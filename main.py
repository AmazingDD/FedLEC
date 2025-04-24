import time
import math
import copy
import random
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import Counter
from sklearn.cluster import KMeans

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.laplace import Laplace

from utils import *
from sampler import partition_data

from models.vit import SViT
from models.resnet import Resnet
from models.vgg import VGG, SimpleCNN, VGGEncoder, VGGClassifier


parser = argparse.ArgumentParser()
# common settings
parser.add_argument('-data_dir', type=str, default='.', help='dataset path')
parser.add_argument('-log_dir', type=str, required=False, default="./logs/", help='Log directory path')
parser.add_argument('-result_dir', type=str, required=False, default="./saved/", help='Model directory path')
parser.add_argument('-model', type=str, default='vgg9', help='neural network used in training')
parser.add_argument('-snn', action='store_true', help="Whether to train SNN or ANN")
parser.add_argument('-dataset', type=str, default='cifar10', help='dataset used for training')
parser.add_argument('-b', '--batch_size', type=int, default=64, help='input batch size for training (default: 64)')
parser.add_argument('-T', type=int, default=4, help='time step for SNN neuron (default: 4)')
parser.add_argument('-lr', type=float, default=1e-3, help='learning rate (default: 0.001)')
parser.add_argument('-optimizer', type=str, default='adam', help='the optimizer')
parser.add_argument('-wd', '--weight_decay', type=float, default=1e-5, help='weight decay for optimizer')
parser.add_argument('-momentum', type=float, default=0.9, help='Parameter controlling the momentum SGD')
parser.add_argument('-gpu', type=int, default=0, help="GPU ID, -1 for CPU")
parser.add_argument('-seed', '--init_seed', type=int, default=2025, help='Random seed')
parser.add_argument('-desc', type=str, default='', help='description for log files')
# federate settings
parser.add_argument('-not_same_initial', action='store_true', help='Whether initial all the models with the same parameters')
parser.add_argument('-noise', type=float, default=0, help='how much noise we add to some party')  # 0.1
parser.add_argument('-noise_type', type=str, default='level', help='Different level or space of noise (Optional: level, space)')
parser.add_argument('-partition', type=str, default='iid', help='the data partitioning strategy')
parser.add_argument('-alpha', type=float, default=0.5, help='The parameter for the dirichlet distribution for data partitioning')
parser.add_argument('-strategy', type=str, default='fedavg', help='fl algorithms: fedavg/fedprox/scaffold/fednova/moon')
parser.add_argument('-np', '--n_parties', type=int, default=10,  help='number of workers in a distributed cluster')
parser.add_argument('-frac', type=float, default=1., help='Sample ratio [0., 1.] of parties for each local training round')
parser.add_argument('-global_epochs', type=int, default=50, help='rounds of updating global model')
parser.add_argument('-local_epochs', type=int, default=10, help='number of local training rounds')
# extra federate settings
parser.add_argument('-mu', type=float, default=0.01, help='the regularization parameter for fedprox')
parser.add_argument('-nc', '--n_clusters', type=int, default=5, help='Number of clusters for FedConcat')
parser.add_argument('-tune_epochs', type=int, default=200, help='Classifier communication round for FedConcat') # 1000 if you want
# parser.add_argument('-eps', type=float, default=0, help='Epsilon for differential privacy to protect label distribution')
parser.add_argument('-lc', action='store_true', help="Whether to do loss calibration tackling label skew with fedavg")
parser.add_argument('-tau', type=float, default=0.01, help='calibration loss constant for fedavg with LC')
parser.add_argument('-rs', action='store_true', help="Whether to do restricted softmax tackling label skew with fedavg")
parser.add_argument('-lamda', type=float, default=0.1, help="Regularization weight for fedLEC")
parser.add_argument('-strength', type=float, default=0.5, help='restricted strength for fedavg with RS')
# extra model settings
parser.add_argument('-nh', '--num_heads', type=int, default=8, help='number of attention heads for spikeformer')
parser.add_argument('-dim', '--hidden_dim', type=int, default=384, help='dimension of intermediate layer in spikeformer FFN part')

args = parser.parse_args()

args.num_classes = CLASS_NUM[args.dataset]
args.device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
if args.device != 'cpu':
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
device = torch.device(args.device)
repeat_num = 0 if args.dataset in ('cifar10-dvs', 'dvs128gesture', 'nmnist', 'ncaltech101') else args.T

# reproducibility
seed = args.init_seed
np.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)

logger = Logger(args, args.desc)
logger.info(str(args))

logger.info("Loading data")
X_train, y_train, X_test, y_test = load_data(args)

logger.info("Partitioning data")
net_dataidx_map = partition_data(args, y_train)

traindata_cls_counts = {}
for net_i, dataidx in net_dataidx_map.items():
    unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
    tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
    traindata_cls_counts[net_i] = tmp
logger.info(f'Finish partitioning, data statistics: {str(traindata_cls_counts)}')

client_distribution = np.zeros((args.n_parties, args.num_classes))
for i in range(args.n_parties):
    for j in traindata_cls_counts[i].keys():
        client_distribution[i][j] = traindata_cls_counts[i][j] / len(net_dataidx_map[i])

train_all_in_list = [] # used to store local train dataset
test_all_in_list = [] # used to store local test dataset
noise_level = 0 # init noise level
for party_id in range(args.n_parties):
    dataidxs = net_dataidx_map[party_id]
    noise_level = args.noise

    if party_id == args.n_parties - 1:
        noise_level = 0 # reset

    if args.noise == 0: # no noise
        train_dl_local, test_dl_local, train_ds_local, test_ds_local = get_dataloader(
            (X_train, X_test), (y_train, y_test), args.batch_size, 32, dataidxs, 0, data_name=args.dataset)
    else:
        # in any case when noise=0 they all will return all local data are sampled without noise
        if args.noise_type == 'space': # add noise to certain pixel points, image space noise
            train_dl_local, test_dl_local, train_ds_local, test_ds_local = get_dataloader(
                (X_train, X_test), (y_train, y_test), args.batch_size, 32, dataidxs, noise_level, party_id, args.n_parties - 1, data_name=args.dataset)
        else:  # noise-based feature imbalance, level
            noise_level = args.noise / (args.n_parties - 1) * party_id
            train_dl_local, test_dl_local, train_ds_local, test_ds_local = get_dataloader(
                (X_train, X_test), (y_train, y_test), args.batch_size, 32, dataidxs, noise_level, data_name=args.dataset)
        
    train_all_in_list.append(train_ds_local)
    test_all_in_list.append(test_ds_local)

train_ds_global = torch.utils.data.ConcatDataset(train_all_in_list)
train_dl_global = torch.utils.data.DataLoader(dataset=train_ds_global, batch_size=args.batch_size, shuffle=True)
test_ds_global = torch.utils.data.ConcatDataset(test_all_in_list)
test_dl_global = torch.utils.data.DataLoader(dataset=test_ds_global, batch_size=32, shuffle=False)
logger.info(f'length of train_dl_global: {len(train_ds_global)}')

logger.info('Inint models')
def init_nets(n_parties, args):
    nets = {net_i: None for net_i in range(n_parties)}

    for net_i in range(n_parties):
        if 'vgg' in args.model:
            net = VGG(args)
        elif 'cnn' in args.model:
            net = SimpleCNN(args)
        elif 'resnet' in args.model:
            net = Resnet(args)
        elif 'vit' in args.model: 
            net = SViT(args)
        else:
            raise NotImplementedError(f'Unkown model name: {args.model}')
        
        nets[net_i] = net

    model_meta_data = []
    layer_type = []
    for k, v in nets[0].state_dict().items():
        model_meta_data.append(v.shape)
        layer_type.append(k)

    return nets, model_meta_data, layer_type

global_acc_record = []

if args.strategy == 'fedavg':
    nets, local_model_meta_data, layer_type = init_nets(args.n_parties, args)
    global_models, global_model_meta_data, global_layer_type = init_nets(1, args)
    global_model = global_models[0]

    global_param = global_model.state_dict()

    if not args.not_same_initial:
        for _, net in nets.items():
            net.load_state_dict(global_param)

    global_model.to(device)

    max_acc = -1.
    for epoch in range(args.global_epochs):
        global_start_time = time.time()
        arr = np.arange(args.n_parties)
        np.random.shuffle(arr)
        selected = arr[:int(args.n_parties * args.frac)]

        global_param = global_model.state_dict()

        for idx in selected:
            nets[idx].load_state_dict(global_param)

        # fedavg local training
        for net_id, net in nets.items():
            if net_id not in selected:
                continue

            net.to(device)
            train_ds_local = train_all_in_list[net_id]
            train_dl_local = torch.utils.data.DataLoader(dataset=train_ds_local, batch_size=args.batch_size, shuffle=True)
            logger.info(f'Training network {net_id}. n_training: {len(train_ds_local)}')

            if args.optimizer == 'adam':
                optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr, weight_decay=args.weight_decay)
            elif args.optimizer == 'sgd':
                optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
            else:
                raise NotImplementedError(f'Not support {args.optimizer}')
            criterion = nn.CrossEntropyLoss().to(device)

            for l_epoch in range(args.local_epochs):
                local_start_time = time.time()
                epoch_loss_collector = []
                for x, target in train_dl_local:
                    x, target = x.to(device), target.to(device)

                    optimizer.zero_grad()
                    x.requires_grad = True
                    target.requires_grad = False
                    target = target.long()

                    # event-driven data, repeat num is 0
                    if repeat_num:
                        x = x.unsqueeze(0).repeat(args.T, 1, 1, 1, 1) # -> (T, B, C, H, W)
                    else: # dvs data
                        x = x.transpose(0, 1) # (B, T, C, H, W) -> (T, B, C, H, W)

                    out = net(x)

                    if args.lc: # make loss calibration
                        local_c_num = [traindata_cls_counts[net_id][c] if c in traindata_cls_counts[net_id].keys() else 0 for c in range(args.num_classes)]
                        for c in range(len(local_c_num)):
                            if local_c_num[c] == 0:
                                out[:, c] -= 10000
                            else:
                                out[:, c] -= math.pow(local_c_num[c], -1 / 4) * args.tau
                    elif args.rs: # make restricted softmax
                        local_c_num = [traindata_cls_counts[net_id][c] if c in traindata_cls_counts[net_id].keys() else 0 for c in range(args.num_classes)]
                        for c in range(len(local_c_num)):
                            if local_c_num[c] == 0:
                                out[:, c] *= args.strength # args.strength
                    else:
                        pass

                    loss = criterion(out, target)

                    loss.backward()
                    optimizer.step()

                    epoch_loss_collector.append(loss.item())

                    functional.reset_net(net)

                epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
                if (1 + l_epoch) % 5 == 0:
                    logger.info(f'Local Epoch [{1 + l_epoch}/{args.local_epochs}] - Loss: {epoch_loss:.4f}, Time elapse: {time.time() - local_start_time:.2f}s')

            train_acc = compute_accuracy(net, train_dl_local, device, repeat_num)
            test_acc = compute_accuracy(net, test_dl_global, device, repeat_num)
            net.to('cpu')

            logger.info(f"net {net_id} Training accuracy: {train_acc:.4f}, Test accuracy: {test_acc:.4f}")

        # global updating
        total_data_points = sum([len(net_dataidx_map[r]) for r in selected])
        fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in selected]

        for idx in range(len(selected)):
            net_param = nets[selected[idx]].cpu().state_dict()
            if idx == 0:
                for key in net_param:
                    global_param[key] = net_param[key] * fed_avg_freqs[idx]
            else:
                for key in net_param:
                    global_param[key] += net_param[key] * fed_avg_freqs[idx]

        global_model.load_state_dict(global_param)
        train_acc = compute_accuracy(global_model, train_dl_global, device, repeat_num)
        test_acc = compute_accuracy(global_model, test_dl_global, device, repeat_num)

        global_acc_record.append(test_acc)
        if max_acc <= test_acc:
            max_acc = test_acc

        logger.info(f"Epoch [{epoch + 1}/{args.global_epochs}] - Global Training accuracy: {train_acc:.4f}, Test accuracy: {test_acc:.4f}, Best test accuracy: {max_acc:.4f}, Train elapse: {time.time() - global_start_time:.2f}s")

elif args.strategy == 'fedprox':
    nets, local_model_meta_data, layer_type = init_nets(args.n_parties, args)
    global_models, global_model_meta_data, global_layer_type = init_nets(1, args)
    global_model = global_models[0]

    global_param = global_model.state_dict()

    if not args.not_same_initial:
        for _, net in nets.items():
            net.load_state_dict(global_param)

    global_model.to(device)

    max_acc = -1.
    for epoch in range(args.global_epochs):
        global_start_time = time.time()

        arr = np.arange(args.n_parties)
        np.random.shuffle(arr)
        selected = arr[:int(args.n_parties * args.frac)]

        global_param = global_model.state_dict()

        for idx in selected:
            nets[idx].load_state_dict(global_param)

        # fedprox local training
        for net_id, net in nets.items():
            if net_id not in selected:
                continue
            dataidxs = net_dataidx_map[net_id]

            net.to(device)
            train_ds_local = train_all_in_list[net_id]
            train_dl_local = torch.utils.data.DataLoader(dataset=train_ds_local, batch_size=args.batch_size, shuffle=True)
            logger.info(f'Training network {net_id}. n_training: {len(train_ds_local)}')

            if args.optimizer == 'adam':
                optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr, weight_decay=args.weight_decay)
            elif args.optimizer == 'sgd':
                optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
            else:
                raise NotImplementedError(f'Not support {args.optimizer}')
            criterion = nn.CrossEntropyLoss().to(device)

            global_weight_collector = list(global_model.to(device).parameters())
            for l_epoch in range(args.local_epochs):
                local_start_time = time.time()
                epoch_loss_collector = []
                for x, target in train_dl_local:
                    x, target = x.to(device), target.to(device)

                    optimizer.zero_grad()
                    x.requires_grad = True
                    target.requires_grad = False
                    target = target.long()

                    if repeat_num:
                        x = x.unsqueeze(0).repeat(args.T, 1, 1, 1, 1) # ->(T, B, C, H, W)
                    else: # dvs data
                        x = x.transpose(0, 1) # (B, T, C, H, W) -> (T, B, C, H, W)

                    out = net(x)

                    loss = criterion(out, target)

                    # special for fedprox
                    fed_prox_reg = 0.
                    for param_index, param in enumerate(net.parameters()):
                        fed_prox_reg += ((args.mu / 2) * torch.norm((param - global_weight_collector[param_index])) ** 2)
                    loss += fed_prox_reg

                    loss.backward()
                    optimizer.step()

                    epoch_loss_collector.append(loss.item())

                    functional.reset_net(net)

                epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
                if (1 + l_epoch) % 5 == 0:
                    logger.info(f'Local Epoch [{1 + l_epoch}/{args.local_epochs}] - Loss: {epoch_loss:.4f}, Time elapse: {time.time() - local_start_time:.2f}s')

            train_acc = compute_accuracy(net, train_dl_local, device, repeat_num)
            test_acc = compute_accuracy(net, test_dl_global, device, repeat_num)
            net.to('cpu')

            logger.info(f'net {net_id} Training accuracy: {train_acc:.4f}, Test accuracy: {test_acc:.4f}')

        # global updating
        total_data_points = sum(len(net_dataidx_map[r]) for r in selected)
        fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in selected]

        for idx in range(len(selected)):
            net_param = nets[selected[idx]].cpu().state_dict()
            if idx == 0:
                for key in net_param:
                    global_param[key] = net_param[key] * fed_avg_freqs[idx]
            else:
                for key in net_param:
                    global_param[key] += net_param[key] * fed_avg_freqs[idx]

        global_model.load_state_dict(global_param)
        train_acc = compute_accuracy(global_model, train_dl_global, device, repeat_num)
        test_acc = compute_accuracy(global_model, test_dl_global, device, repeat_num)

        global_acc_record.append(test_acc)
        if max_acc <= test_acc:
            max_acc = test_acc
        
        logger.info(f"Epoch [{epoch + 1}/{args.global_epochs}] - Global Training accuracy: {train_acc:.4f}, Test accuracy: {test_acc:.4f}, Best test accuracy: {max_acc:.4f}, Train elapse: {time.time() - global_start_time:.2f}s")

elif args.strategy == 'scaffold':
    nets, local_model_meta_data, layer_type = init_nets(args.n_parties, args)
    global_models, global_model_meta_data, global_layer_type = init_nets(1, args)
    global_model = global_models[0]
    global_param = global_model.state_dict()

    c_nets, _, _ = init_nets(args.n_parties, args)
    c_globals, _, _ = init_nets(1, args)
    c_global = c_globals[0]
    c_global_param = c_global.state_dict()

    for net_id, net in c_nets.items():
        net.load_state_dict(c_global_param)

    if not args.not_same_initial:
        for _, net in nets.items():
            net.load_state_dict(global_param)

    global_model.to(device)
    c_global.to(device)

    max_acc = -1.
    for epoch in range(args.global_epochs):
        global_start_time = time.time()

        arr = np.arange(args.n_parties)
        np.random.shuffle(arr)
        selected = arr[:int(args.n_parties * args.frac)]

        global_param = global_model.state_dict()

        for idx in selected:
            nets[idx].load_state_dict(global_param)

        # scaffold local training
        total_delta = copy.deepcopy(global_model.state_dict())
        for key in total_delta:
            total_delta[key] = 0.

        for net_id, net in nets.items():
            if net_id not in selected:
                continue

            net.to(device)
            c_nets[net_id].to(device)

            train_ds_local = train_all_in_list[net_id]
            train_dl_local = torch.utils.data.DataLoader(dataset=train_ds_local, batch_size=args.batch_size, shuffle=True)
            logger.info(f'Training network {net_id}. n_training: {len(train_ds_local)}')

            if args.optimizer == 'adam':
                optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr, weight_decay=args.weight_decay)
            elif args.optimizer == 'sgd':
                optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
            else:
                raise NotImplementedError(f'Not support {args.optimizer}')
            criterion = nn.CrossEntropyLoss().to(device)

            c_global_param = c_global.state_dict()
            c_local_param = c_nets[net_id].state_dict()
            
            tau = 0
            for l_epoch in range(args.local_epochs):
                local_start_time = time.time()
                epoch_loss_collector = []
                for x, target in train_dl_local:
                    x, target = x.to(device), target.to(device)

                    optimizer.zero_grad()
                    x.requires_grad = True
                    target.requires_grad = False
                    target = target.long()

                    if repeat_num:
                        x = x.unsqueeze(0).repeat(args.T, 1, 1, 1, 1) # -> (T, B, C, H, W)
                    else: # dvs data
                        x = x.transpose(0, 1) # (B, T, C, H, W) -> (T, B, C, H, W)

                    out = net(x)
                    loss = criterion(out, target)

                    loss.backward()
                    optimizer.step()

                    net_param = net.state_dict()
                    for key in net_param:
                        net_param[key] = net_param[key] - args.lr * (c_global_param[key] - c_local_param[key]) # -c + c_i^t
                    net.load_state_dict(net_param)

                    tau += 1
                    epoch_loss_collector.append(loss.item())

                    functional.reset_net(net)

                epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
                if (1 + l_epoch) % 5 == 0:
                    logger.info(f'Local Epoch [{1 + l_epoch}/{args.local_epochs}] - Loss: {epoch_loss:.4f}, Time elapse: {time.time() - local_start_time:.2f}s')

            c_new_param = c_nets[net_id].state_dict()
            c_delta_param = copy.deepcopy(c_nets[net_id].state_dict())
            global_model_param = global_model.state_dict()
            net_param = net.state_dict()
            for key in net_param:
                c_new_param[key] = c_new_param[key] - c_global_param[key] + (global_model_param[key] - net_param[key]) / (tau * args.lr) # c_i - c + \frac{1}{\tau * lr}(w^t - w_i^t)
                c_delta_param[key] = c_new_param[key] - c_local_param[key]
            c_nets[net_id].load_state_dict(c_new_param)

            train_acc = compute_accuracy(net, train_dl_local, device, repeat_num)
            test_acc = compute_accuracy(net, test_dl_global, device, repeat_num)
            
            net.to('cpu')
            c_nets[net_id].to('cpu')

            for key in total_delta:
                total_delta[key] += c_delta_param[key] # \delta c = c_i* - c_i

            logger.info(f"net {net_id} Training accuracy: {train_acc:.4f}, Test accuracy: {test_acc:.4f}")

        for key in total_delta:
            total_delta[key] /= args.n_parties
        c_global_param = c_global.state_dict()
        for key in c_global_param:
            if c_global_param[key].type() == 'torch.LongTensor':
                c_global_param[key] += total_delta[key].type(torch.LongTensor)
            elif c_global_param[key].type() == 'torch.cuda.LongTensor':
                c_global_param[key] += total_delta[key].type(torch.cuda.LongTensor)
            else:
                c_global_param[key] += total_delta[key]
        c_global.load_state_dict(c_global_param)

        # global updating
        total_data_points = sum([len(net_dataidx_map[r]) for r in selected])
        fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in selected]

        for idx in range(len(selected)):
            net_para = nets[selected[idx]].cpu().state_dict()
            if idx == 0:
                for key in net_para:
                    global_param[key] = net_para[key] * fed_avg_freqs[idx]
            else:
                for key in net_para:
                    global_param[key] += net_para[key] * fed_avg_freqs[idx]

        global_model.load_state_dict(global_param)
        train_acc = compute_accuracy(global_model, train_dl_global, device, repeat_num)
        test_acc = compute_accuracy(global_model, test_dl_global, device, repeat_num)

        global_acc_record.append(test_acc)
        if max_acc <= test_acc:
            max_acc = test_acc

        logger.info(f"Epoch [{epoch + 1}/{args.global_epochs}] - Global Training accuracy: {train_acc:.4f}, Test accuracy: {test_acc:.4f}, Best test accuracy: {max_acc:.4f}, Train elapse: {time.time() - global_start_time:.2f}s")

elif args.strategy == 'fednova':
    nets, local_model_meta_data, layer_type = init_nets(args.n_parties, args)
    global_models, global_model_meta_data, global_layer_type = init_nets(1, args)
    global_model = global_models[0]

    d_list = [copy.deepcopy(global_model.state_dict()) for i in range(args.n_parties)]
    d_total_round = copy.deepcopy(global_model.state_dict())
    for i in range(args.n_parties):
        for key in d_list[i]:
            d_list[i][key] = 0
    for key in d_total_round:
        d_total_round[key] = 0

    global_param = global_model.state_dict()

    if not args.not_same_initial:
        for _, net in nets.items():
            net.load_state_dict(global_param)

    max_acc = -1.
    for epoch in range(args.global_epochs):
        global_start_time = time.time()
        arr = np.arange(args.n_parties)
        np.random.shuffle(arr)
        selected = arr[:int(args.n_parties * args.frac)]

        global_param = global_model.state_dict()

        for idx in selected:
            nets[idx].load_state_dict(global_param)

        # fednova local training
        a_list, d_list, n_list = [], [], []
        global_model.to(device)

        for net_id, net in nets.items():
            if net_id not in selected:
                continue

            net.to(device)
            train_ds_local = train_all_in_list[net_id]

            train_dl_local = torch.utils.data.DataLoader(dataset=train_ds_local, batch_size=args.batch_size, shuffle=True)
            logger.info(f'Training network {net_id}. n_training: {len(train_ds_local)}')

            optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
            criterion = nn.CrossEntropyLoss().to(device)

            tau = 0
            for l_epoch in range(args.local_epochs):
                local_start_time = time.time()
                epoch_loss_collector = []
                for x, target in train_dl_local:

                    x, target = x.to(device), target.to(device)

                    optimizer.zero_grad()
                    x.requires_grad = True
                    target.requires_grad = False
                    target = target.long()

                    # event-driven data, repeat num is 0
                    if repeat_num:
                        x = x.unsqueeze(0).repeat(args.T, 1, 1, 1, 1) # -> (T, B, C, H, W)
                    else: # dvs data
                        x = x.transpose(0, 1) # (B, T, C, H, W) -> (T, B, C, H, W)

                    out = net(x)                    
                    loss = criterion(out, target)

                    loss.backward()
                    optimizer.step()

                    tau += 1

                    epoch_loss_collector.append(loss.item())

                    functional.reset_net(net)

                epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
                if (1 + l_epoch) % 5 == 0:
                    logger.info(f'Local Epoch [{1 + l_epoch}/{args.local_epochs}] - Loss: {epoch_loss:.4f}, Time elapse: {time.time() - local_start_time:.2f}s')

            a_i = (tau - args.momentum * (1 - pow(args.momentum, tau)) / (1 - args.momentum)) / (1 - args.momentum)

            global_model_param = global_model.state_dict()
            net_param = net.state_dict()
            norm_grad = copy.deepcopy(global_model.state_dict())

            for key in norm_grad:
                norm_grad[key] = torch.true_divide(global_model_param[key] - net_param[key], a_i)

            train_acc = compute_accuracy(net, train_dl_local, device, repeat_num)
            test_acc = compute_accuracy(net, test_dl_global, device, repeat_num)
            net.to('cpu')

            a_list.append(a_i)
            d_list.append(norm_grad)
            n_i = len(train_dl_local.dataset)
            n_list.append(n_i)

            logger.info(f"net {net_id} Training accuracy: {train_acc:.4f}, Test accuracy: {test_acc:.4f}")

        # global updating  
        total_n = sum(n_list)
        d_total_round = copy.deepcopy(global_model.state_dict())
        for key in d_total_round:
            d_total_round[key] = 0.

        for i in range(len(selected)):
            d_param = d_list[i]
            for key in d_param:
                d_total_round[key] += d_param[key] * n_list[i] / total_n

        coef = 0.0
        for i in range(len(selected)):
            coef = coef + a_list[i] * n_list[i] / total_n

        updated_model = global_model.state_dict()
        for key in updated_model:
            if updated_model[key].type() == 'torch.LongTensor':
                updated_model[key] -= (coef * d_total_round[key]).type(torch.LongTensor)
            elif updated_model[key].type() == 'torch.cuda.LongTensor':
                updated_model[key] -= (coef * d_total_round[key]).type(torch.cuda.LongTensor)
            else:
                updated_model[key] -= coef * d_total_round[key]

        global_model.load_state_dict(updated_model)
        train_acc = compute_accuracy(global_model, train_dl_global, device, repeat_num)
        test_acc = compute_accuracy(global_model, test_dl_global, device, repeat_num)

        global_acc_record.append(test_acc)
        if max_acc <= test_acc:
            max_acc = test_acc

        logger.info(f"Epoch [{epoch + 1}/{args.global_epochs}] - Global Training accuracy: {train_acc:.4f}, Test accuracy: {test_acc:.4f}, Best test accuracy: {max_acc:.4f}, Train elapse: {time.time() - global_start_time:.2f}s")

elif args.strategy == 'raw':
    # normal training
    global_models, global_model_meta_data, global_layer_type = init_nets(1, args)
    global_model = global_models[0]

    global_model.to(device)

    if args.optimizer == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, global_model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, global_model.parameters()), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    else:
        raise NotImplementedError(f'Not support {args.optimizer}')
    criterion = nn.CrossEntropyLoss().to(device)

    max_acc = -1.
    for epoch in range(args.global_epochs):
        global_start_time = time.time()

        for x, target in tqdm(train_dl_global, desc='training', unit='batch'): 
            x, target = x.to(device), target.to(device)

            optimizer.zero_grad()
            x.requires_grad = True
            target.requires_grad = False
            target = target.long()

            if repeat_num:
                x = x.unsqueeze(0).repeat(args.T, 1, 1, 1, 1) # -> (T, B, C, H, W)
            else: # dvs data
                x = x.transpose(0, 1) # (B, T, C, H, W) -> (T, B, C, H, W)
            
            out = global_model(x)
            loss = criterion(out, target)

            loss.backward()
            optimizer.step()

            functional.reset_net(global_model)

        train_acc = compute_accuracy(global_model, train_dl_global, device, repeat_num)
        test_acc = compute_accuracy(global_model, test_dl_global, device, repeat_num)


        global_acc_record.append(test_acc)
        if max_acc <= test_acc:
            max_acc = test_acc
    
        logger.info(f"Epoch [{epoch + 1}/{args.global_epochs}] - Global Training accuracy: {train_acc:.4f}, Test accuracy: {test_acc:.4f}, Best test accuracy: {max_acc:.4f}, Train elapse: {time.time() - global_start_time:.2f}s")

    global_param = global_model.state_dict()

elif args.strategy == 'fedconcat':
    class CombineModels(nn.Module):
        def __init__(self, net1, net2):
            super(CombineModels, self).__init__()

            self.encode = net1
            self.cls = net2

        def forward(self, x):
            x = self.cls(self.encode(x))
            return x

    class CombineAllEncoder(nn.Module):
        def __init__(self, encoder_list):
            super(CombineAllEncoder, self).__init__()

            self.nets = nn.ModuleList(encoder_list)

        def forward(self, x):
            n = len(self.nets)
            features = self.nets[0](x)
            for i in range(1, n):
                features = torch.cat((features, self.nets[i](x)), 2)

            return features  # (T, B, D)


    encoder_list, classifier_list = [], []
    for i in range(args.n_parties):
        encoder = VGGEncoder(args)
        encoder_list.append(encoder)

        classifier = VGGClassifier(args)  
        classifier_list.append(classifier)

    num_k = args.n_clusters
    encoder_global_param = [encoder_list[0].state_dict() for i in range(num_k)]
    classifier_global_param = [classifier_list[0].state_dict() for i in range(num_k)]

    estimator = KMeans(n_clusters=num_k)
    estimator.fit(client_distribution)
    assign = estimator.labels_
    
    logger.info(assign)
    group = [[] for i in range(num_k)]
    for i in range(args.n_parties):
        group[assign[i]].append(i)
    logger.info(group)

    for e_epoch in range(args.global_epochs):
        logger.info(f'Start Encoder training epoch [{e_epoch + 1}/{args.global_epochs}]')
        top = int(args.n_parties * args.frac)
        participation = np.random.permutation(np.arange(args.n_parties))[:top]

        for i in range(num_k):
            selected = []
            for x in group[i]:
                if x in participation:
                    selected.append(x)
            if len(selected) == 0:
                continue

            for idx in selected:
                encoder_list[idx].load_state_dict(encoder_global_param[i])
                classifier_list[idx].load_state_dict(classifier_global_param[i])


            for net_id in range(len(encoder_list)):
                if net_id not in selected:
                    continue

                dataidxs = net_dataidx_map[net_id]
            
                encoder_list[net_id].to(device)
                classifier_list[net_id].to(device)

                train_ds_local = train_all_in_list[net_id]
                train_dl_local = torch.utils.data.DataLoader(dataset=train_ds_local, batch_size=args.batch_size, shuffle=True)
                logger.info(f'Training network {net_id}. n_training: {len(train_ds_local)}')

                if args.optimizer == 'adam':
                    optimizer = optim.Adam([{'params': encoder_list[net_id].parameters()}, {'params': classifier_list[net_id].parameters()}], lr=args.lr, weight_decay=args.weight_decay)
                elif args.optimizer == 'sgd':
                    wd_factor = args.weight_decay
                    if args.dataset in ("cifar100", "tinyimagenet"):
                        if args.partition == 'noniid-labeldir':
                            if args.dataset == 'cifar100' and args.beta > 0.4:
                                wd_factor = 0.005
                            else:
                                wd_factor = 0.002
                        else:
                            wd_factor = 0.001
                    optimizer = optim.SGD([{'params': encoder_list[net_id].parameters()}, {'params': classifier_list[net_id].parameters()}], lr=args.lr, momentum=args.momentum, weight_decay=wd_factor)
                else:
                    raise NotImplementedError(f'Not support {args.optimizer}')
                classification_loss = nn.CrossEntropyLoss().to(device)

                for l_epoch in range(args.local_epochs):
                    local_start_time = time.time()
                    epoch_loss_collector = []
                    for x, target in train_dl_local:
                        x, target = x.to(device), target.to(device)

                        optimizer.zero_grad()
                        x.requires_grad = True
                        target.requires_grad = False
                        target = target.long()

                        # event-driven data, repeat num is 0
                        if repeat_num:
                            x = x.unsqueeze(0).repeat(args.T, 1, 1, 1, 1) # -> (T, B, C, H, W)
                        else: # dvs data
                            x = x.transpose(0, 1) # (B, T, C, H, W) -> (T, B, C, H, W)

                        out = encoder_list[net_id](x)
                        out = classifier_list[net_id](out)

                        loss = classification_loss(out, target)

                        loss.backward()
                        optimizer.step()
                        epoch_loss_collector.append(loss.item())

                        functional.reset_net(encoder_list[net_id])

                    epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
                    if (1 + l_epoch) % 5 == 0:
                        logger.info(f'Local Epoch [{1 + l_epoch}/{args.local_epochs}] - Loss: {epoch_loss:.4f}, Time elapse: {time.time() - local_start_time:.2f}s')
                
                encoder_list[net_id].to('cpu')
                classifier_list[net_id].to('cpu')

            total_data_points = sum([len(net_dataidx_map[r]) for r in selected])
            fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in selected]

            for idx in range(len(selected)):
                net_param = encoder_list[selected[idx]].cpu().state_dict()
                if idx == 0:
                    for key in net_param:
                        encoder_global_param[i][key] = net_param[key] * fed_avg_freqs[idx]
                else:
                    for key in net_param:
                        encoder_global_param[i][key] += net_param[key] * fed_avg_freqs[idx]

            for idx in range(len(selected)):
                net_param = classifier_list[selected[idx]].cpu().state_dict()
                if idx == 0:
                    for key in net_param:
                        classifier_global_param[i][key] = net_param[key] * fed_avg_freqs[idx]
                else:
                    for key in net_param:
                        classifier_global_param[i][key] += net_param[key] * fed_avg_freqs[idx]

    encoder_selected = []
    for i in range(num_k):
        encoder_selected.append(encoder_list[i])
        encoder_selected[i].load_state_dict(encoder_global_param[i])
        encoder_selected[i].to(device)
    classifier_selected = []
    for i in range(num_k):
        classifier_selected.append(classifier_list[i])
        classifier_selected[i].load_state_dict(classifier_global_param[i])
        classifier_selected[i].to(device)

    encoder_all = CombineAllEncoder(encoder_selected)
    larger_classifier_global = VGGClassifier(args, num_k)

    # if set to True, the final classifier layer will be initialized as the previous cluster model's classifier layers (to speed up convergence).
    start_from_classifier_weight = True if args.dataset in ("cifar100", "tinyimagenet") else False
    if start_from_classifier_weight:
        # Retrieve the weights from each model in classifier_global_param
        weights = [model.fc.weight for model in classifier_selected]
        biases = [model.fc.bias for model in classifier_selected] 
        # Concatenate along the dimension of number of neurons in the hidden layer
        # The weights are transposed before concatenating because in PyTorch, the Linear layer's weight shape is (output_dim, input_dim)
        weights_tensor = torch.cat([w.t() for w in weights], dim=0).t()
        biases_tensor = sum(biases)
    
        logger.info(weights_tensor.shape)
        logger.info(biases_tensor.shape)

        # Create a state dictionary to load into the larger_classifier_global
        state_dict = {
            'fc.weight': weights_tensor,
            'fc.bias': biases_tensor,
        }

        # Load the state dict into the model
        larger_classifier_global.load_state_dict(state_dict)
        logger.info("loading initial classifier parameters done")

    larger_classifier_global_param = larger_classifier_global.cpu().state_dict()

    larger_classifier_list = []
    for t in range(args.n_parties):
        larger_classifier_list.append(VGGClassifier(args, num_k))

    global_param = None
    
    max_acc = -1.
    for epoch in range(args.tune_epochs):
        global_start_time = time.time()

        arr = np.arange(args.n_parties)
        np.random.shuffle(arr)
        selected = arr[:int(args.n_parties * args.frac)]

        for idx in selected:
            larger_classifier_list[idx].load_state_dict(larger_classifier_global_param)

        for net_id in range(len(larger_classifier_list)):
            if net_id not in selected:
                continue

            dataidxs = net_dataidx_map[net_id]

            encoder_all.to(device)
            larger_classifier_list[net_id].to(device)

            train_ds_local = train_all_in_list[net_id]
            train_dl_local = torch.utils.data.DataLoader(dataset=train_ds_local, batch_size=args.batch_size, shuffle=True)

            if args.optimizer == 'adam':
                optimizer = optim.Adam([{'params': larger_classifier_list[net_id].parameters()}], lr=args.lr, weight_decay=args.weight_decay)
            elif args.optimizer == 'sgd':
                optimizer = optim.SGD([{'params': larger_classifier_list[net_id].parameters()}], lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
            criterion = nn.CrossEntropyLoss().to(device)

            cnt = 0
            for x, target in train_dl_local:
                x, target = x.to(device), target.to(device)

                optimizer.zero_grad()
                x.requires_grad = True
                target.requires_grad = False
                target = target.long()

                if repeat_num:
                    x = x.unsqueeze(0).repeat(args.T, 1, 1, 1, 1)
                else:
                    x = x.transpose(0, 1)

                out = larger_classifier_list[net_id](encoder_all(x))
                loss = criterion(out, target)

                loss.backward()
                optimizer.step()

                functional.reset_net(encoder_all)

                cnt += 1
                if cnt == 3:
                    break # only train 3 mini-batches

        logger.info(f' **Epoch {epoch + 1} Fine-tuning complete **')
        # This is because we move 3 steps for all clients, so this average should be unweighted.
        fed_avg_freqs = [1 / len(selected) for r in selected]  

        for idx in range(len(selected)):
            net_param = larger_classifier_list[selected[idx]].cpu().state_dict()
            if idx == 0:
                for key in net_param:
                    larger_classifier_global_param[key] = net_param[key] * fed_avg_freqs[idx]
            else:
                for key in net_param:
                    larger_classifier_global_param[key] += net_param[key] * fed_avg_freqs[idx]

        larger_classifier_global.load_state_dict(larger_classifier_global_param)

        global_model = CombineModels(encoder_all, larger_classifier_global)
        global_model.to(device)

        test_acc = compute_accuracy(global_model, test_dl_global, device, repeat_num)

        global_acc_record.append(test_acc)
        if max_acc <= test_acc:
            max_acc = test_acc

        if (epoch + 1) % 50 == 0:
            logger.info(f"Epoch [{epoch + 1}/{args.tune_epochs}] - Test accuracy: {test_acc:.4f}, Best test accuracy: {max_acc:.4f}, Train elapse: {time.time() - global_start_time:.2f}s")
    
    global_param = global_model.state_dict()
    args.global_epochs = args.tune_epochs

elif args.strategy == 'flea':
    nets, local_model_meta_data, layer_type = init_nets(args.n_parties, args)
    global_models, global_model_meta_data, global_layer_type = init_nets(1, args)
    global_model = global_models[0]

    global_param = global_model.state_dict()

    if not args.not_same_initial:
        for _, net in nets.items():
            net.load_state_dict(global_param)

    global_model.to(device)

    layer_index = 13
    percent = 0.1
    history_clients = []
    features = {}

    max_acc = -1.
    warmup = 1
    for epoch in range(warmup):
        global_start_time = time.time()
        arr = np.arange(args.n_parties)
        np.random.shuffle(arr)
        # selected = arr[:int(args.n_parties * args.frac)]
        selected = arr[:] # all client participate for warmup

        global_param = global_model.state_dict()

        for idx in selected:
            nets[idx].load_state_dict(global_param)

        for net_id, net in nets.items():
            if net_id not in selected:
                continue
            net.to(device)
            train_ds_local = train_all_in_list[net_id]
            train_dl_local = torch.utils.data.DataLoader(dataset=train_ds_local, batch_size=args.batch_size, shuffle=True)
            logger.info(f'Training network {net_id}. n_training: {len(train_ds_local)}')

            lr_decay = 1 - epoch * 0.018 if epoch < 50 else 0.1
            logger.info(f'Local adapation with distilling: {args.lr * lr_decay}')

            if args.optimizer == 'adam':
                optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr * lr_decay, weight_decay=args.weight_decay)
            elif args.optimizer == 'sgd':
                optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr * lr_decay, momentum=args.momentum, weight_decay=args.weight_decay)
            else:
                raise NotImplementedError(f'Not support {args.optimizer}')
            
            KLLoss = nn.KLDivLoss(reduction='batchmean')
            CELoss = nn.CrossEntropyLoss(reduction='mean')
            CorLoss = Corelation()

            for l_epoch in range(args.local_epochs * 3):
                local_start_time = time.time()
                epoch_loss_collector = []
                for x, target in train_dl_local:
                    x, target = x.to(device), target.to(device)
                    
                    optimizer.zero_grad()
                    x.requires_grad = True
                    target.requires_grad = False
                    target = target.long()

                    # event-driven data, repeat num is 0
                    if repeat_num:
                        x = x.unsqueeze(0).repeat(args.T, 1, 1, 1, 1) # -> (T, B, C, H, W)
                    else: # dvs data
                        x = x.transpose(0, 1) # (B, T, C, H, W) -> (T, B, C, H, W)

                    batch_f = net.get_feature(x, idx=layer_index)
                    functional.reset_net(net)

                    # local data CE
                    out = net(x)
                    loss1 = CELoss(out, target)

                    # local data distilling
                    tau = 1
                    beta = 0.5
                    logit_gb = global_model(x)
                    pro_gb = F.softmax(logit_gb / tau, dim=1)
                    pro_lc = F.log_softmax(out / tau, dim=1)
                    loss2 = beta * (tau ** 2) * KLLoss(pro_lc,pro_gb)

                    # local feature decorrelation
                    alpha3 = 3
                    loss3 = CorLoss(x, batch_f) * alpha3

                    # total loss
                    loss = loss1 + loss2 + loss3 

                    loss.backward()
                    optimizer.step()

                    epoch_loss_collector.append(loss.item())

                    functional.reset_net(net)
                    functional.reset_net(global_model)

                epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
                if (1 + l_epoch) % 5 == 0:
                    logger.info(f'Local Epoch [{1 + l_epoch}/{args.local_epochs * 3}] - Loss: {epoch_loss:.4f}, Time elapse: {time.time() - local_start_time:.2f}s')

            train_acc = compute_accuracy(net, train_dl_local, device, repeat_num)
            test_acc = compute_accuracy(net, test_dl_global, device, repeat_num)
            net.to('cpu')

            logger.info(f"net {net_id} Training accuracy: {train_acc:.4f}, Test accuracy: {test_acc:.4f}")

            history_clients.append(net_id)

        # global updating
        total_data_points = sum([len(net_dataidx_map[r]) for r in selected])
        fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in selected]

        for idx in range(len(selected)):
            net_param = nets[selected[idx]].cpu().state_dict()
            if idx == 0:
                for key in net_param:
                    global_param[key] = net_param[key] * fed_avg_freqs[idx]
            else:
                for key in net_param:
                    global_param[key] += net_param[key] * fed_avg_freqs[idx]

        global_model.load_state_dict(global_param)
        train_acc = compute_accuracy(global_model, train_dl_global, device, repeat_num)
        test_acc = compute_accuracy(global_model, test_dl_global, device, repeat_num)

        global_acc_record.append(test_acc)
        if max_acc <= test_acc:
            max_acc = test_acc

        logger.info(f"Warmup Epoch [{epoch + 1}/{args.global_epochs}] - Global Training accuracy: {train_acc:.4f}, Test accuracy: {test_acc:.4f}, Best test accuracy: {max_acc:.4f}, Train elapse: {time.time() - global_start_time:.2f}s")

        for idx in selected:
            nets[idx].load_state_dict(global_param)

        # update feature buffer
        history_round = 1
        remain_users = history_clients[-args.n_parties * history_round:]
        current_users = [uid for uid in features.keys()]
        for uid in current_users:
            if uid not in remain_users:
                features.pop(uid)
        for idx in selected:
            # send features
            nets[idx].eval()
            train_ds_local = train_all_in_list[idx]
            train_dl_local = torch.utils.data.DataLoader(dataset=train_ds_local, batch_size=args.batch_size, shuffle=True)
            iter_trainloader = iter(train_dl_local)
            n_batch = len(train_ds_local) // args.batch_size + 1 if len(train_ds_local) % args.batch_size else len(train_ds_local) // args.batch_size

            with torch.no_grad():
                # hold_data = self.send_data(fre=1,fraction=percent)
                for batch in range(n_batch):
                    batch_x, batch_y = next(iter_trainloader)
                    if batch == 0:
                        X, Y = batch_x, batch_y
                    else:
                        Y = torch.cat((Y, batch_y), dim=0)
                        X = torch.cat((X, batch_x), dim=0)

                cnt = int(percent * (len(X)))
                sample_idxs = np.random.choice(range(len(X)), cnt).tolist()
                feature = X[sample_idxs, ...]  # B, (C, H, W) / (T, C, H, W)
                label = Y[sample_idxs]

                # event-driven data, repeat num is 0
                if repeat_num:
                    feature = feature.unsqueeze(0).repeat(args.T, 1, 1, 1, 1) # -> (T, B, C, H, W)
                else: # dvs data
                    feature = feature.transpose(0, 1) # (B, T, C, H, W) -> (T, B, C, H, W)

                if layer_index >= 0:
                    feature = nets[idx].get_feature(feature, idx=layer_index)
                    functional.reset_net(nets[idx])
                # else share raw data with negative layers

            features[idx] = [feature, label]
            del feature, label

        logger.info(f'size of the global feature buffer: {len(features)}')
        torch.cuda.empty_cache()

    # send features to each client
    for i, idx in enumerate(features.keys()):
        if i == 0:
            local_features, local_labels = features[idx]
        else:
            feature, label = features[idx] # (T, B, C, H, W)
            local_features = torch.cat((local_features, feature), dim=1) # concat along B-dim
            local_labels = torch.cat((local_labels, label))

    # change (T, B, ...) to (B, T, ...) for dataloader index correctly
    local_features = local_features.transpose(0, 1)

    all_features = [(x, y) for x, y in zip(local_features, local_labels)] 
    logger.info(f'Current data/feature buff: {local_features.shape}, {local_labels.shape}, {len(all_features)}') 
    local_labels = local_labels.to('cpu').tolist()
    logger.info(f'Buffer classes: {Counter(local_labels)}')

    for epoch in range(warmup, args.global_epochs):
        global_start_time = time.time()
        arr = np.arange(args.n_parties)
        np.random.shuffle(arr)
        selected = arr[:int(args.n_parties * args.frac)]

        global_param = global_model.state_dict()

        # send parameters
        for idx in selected:
            nets[idx].load_state_dict(global_param)
        
        for net_id, net in nets.items():
            if net_id not in selected:
                continue
            net.to(device)

            train_ds_local = train_all_in_list[net_id]
            # n_batch = len(train_ds_local) // args.batch_size + 1
            n_batch = len(train_ds_local) // args.batch_size + 1 if len(train_ds_local) % args.batch_size else len(train_ds_local) // args.batch_size
            train_dl_local = torch.utils.data.DataLoader(dataset=train_ds_local, batch_size=args.batch_size, shuffle=True)
            logger.info(f'Training network {net_id}. n_training: {len(train_ds_local)}')

            # feature loader for each selected client
            feature_batch = len(all_features) // args.batch_size + 1 if len(all_features) % args.batch_size else len(all_features) // args.batch_size
            featureloader = DataLoader(all_features, args.batch_size, shuffle=True)
            torch.cuda.empty_cache()
            n_batch = min(n_batch, feature_batch) # local training need features from warmup to support 

            lr_decay = 1 - epoch * 0.018 if epoch < 50 else 0.1
            alpha3 = 3 - epoch * 0.1 if epoch < 20 else 1
            logger.info(f'Local adapation with distilling: lr {args.lr * lr_decay}, alpha3 {alpha3}')

            if args.optimizer == 'adam':
                optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr * lr_decay, weight_decay=args.weight_decay)
            elif args.optimizer == 'sgd':
                optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr * lr_decay, momentum=args.momentum, weight_decay=args.weight_decay)
            else:
                raise NotImplementedError(f'Not support {args.optimizer}')

            net.train()
            global_model.eval()

            tau = 1
            KLLoss = nn.KLDivLoss(reduction='batchmean')
            Loss = nn.CrossEntropyLoss(reduction='mean')
            CorLoss = Corelation()
            MCELoss = Multilabel() 

            for l_epoch in range(args.local_epochs):
                local_start_time = time.time()
                epoch_loss_collector = []

                iter_trainloader = iter(train_dl_local)
                iter_featureloader = iter(featureloader)

                for _ in range(n_batch):
                    optimizer.zero_grad()

                    batch_X, batch_Y = next(iter_trainloader)
                    if repeat_num:
                        batch_X = batch_X.unsqueeze(0).repeat(args.T, 1, 1, 1, 1)
                    else:
                        batch_X = batch_X.transpose(0, 1)

                    batch_X_fea, batch_Y_fea = next(iter_featureloader)

                    batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)
                    batch_X_fea = batch_X_fea.transpose(0, 1).to(device) # (B, T, ...) to (T, B, ...)
                    batch_Y_fea = batch_Y_fea.to(device)

                    batch_F = net.get_feature(batch_X, idx=layer_index)
                    functional.reset_net(net)

                    fea_num = min(batch_X.shape[1], batch_X_fea.shape[1])
                    batch_x = batch_F[:, :fea_num, ...]
                    batch_y = batch_Y[:fea_num,]
                    batch_X_fea = batch_X_fea[:, :fea_num, ...]
                    batch_Y_fea = batch_Y_fea[:fea_num,]

                    lam = np.random.beta(2, 2, fea_num)
                    mix_x = torch.zeros(batch_x.shape).to(device)
                    mix_y = torch.zeros(F.one_hot(batch_y, args.num_classes).shape).to(device)
                    for i in range(fea_num):
                        mix_x[:, i, ...] = lam[i] * batch_x[:, i, ...] + (1-lam[i]) * batch_X_fea[:, i, ...]
                        mix_y[i, :] = lam[i] * F.one_hot(batch_y, args.num_classes)[i, :] + (1 - lam[i]) * F.one_hot(batch_Y_fea, args.num_classes)[i, :]

                    logit = net(batch_X)
                    functional.reset_net(net)
                    loss1 = Loss(logit, batch_Y) 

                    # global data CE
                    mix_logit = net.forward_feature(mix_x, idx=layer_index)
                    functional.reset_net(net)
                    loss2 = MCELoss(mix_logit, mix_y)

                    # local feature decorrelation
                    loss3 = CorLoss(batch_X, batch_F)

                    # global data distilling
                    logit_lc = net.forward_feature(batch_X_fea, idx=layer_index)
                    functional.reset_net(net)
                    logit_gb = global_model.forward_feature(batch_X_fea, idx=layer_index) 
                    functional.reset_net(global_model)
                    pro_gb = F.softmax(logit_gb / tau, dim=1)     ## y
                    pro_lc = F.log_softmax(logit_lc / tau, dim=1) ## x
                    # The smallest KL is 0, always positive 
                    loss4 = KLLoss(pro_lc, pro_gb) * (tau ** 2)

                    loss = loss1 + loss2 + loss3 * alpha3 + loss4

                    loss.backward()
                    optimizer.step()

                    del loss1, loss2, loss3, loss4, logit, logit_gb, logit_lc, pro_gb, pro_lc, batch_X, batch_Y, batch_X_fea, batch_Y_fea

                    epoch_loss_collector.append(loss.item())

                    functional.reset_net(net)
                    functional.reset_net(global_model)

                epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
                if (1 + l_epoch) % 5 == 0:
                    logger.info(f'Local Epoch [{1 + l_epoch}/{args.local_epochs}] - Loss: {epoch_loss:.4f}, Time elapse: {time.time() - local_start_time:.2f}s')

            train_acc = compute_accuracy(net, train_dl_local, device, repeat_num)
            test_acc = compute_accuracy(net, test_dl_global, device, repeat_num)
            net.to('cpu')

            logger.info(f"net {net_id} Training accuracy: {train_acc:.4f}, Test accuracy: {test_acc:.4f}")

            history_clients.append(net_id)

        # global updating
        total_data_points = sum([len(net_dataidx_map[r]) for r in selected])
        fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in selected]

        for idx in range(len(selected)):
            net_param = nets[selected[idx]].cpu().state_dict()
            if idx == 0:
                for key in net_param:
                    global_param[key] = net_param[key] * fed_avg_freqs[idx]
            else:
                for key in net_param:
                    global_param[key] += net_param[key] * fed_avg_freqs[idx]

        global_model.load_state_dict(global_param)
        train_acc = compute_accuracy(global_model, train_dl_global, device, repeat_num)
        test_acc = compute_accuracy(global_model, test_dl_global, device, repeat_num)

        global_acc_record.append(test_acc)
        if max_acc <= test_acc:
            max_acc = test_acc

        logger.info(f"Epoch [{epoch + 1}/{args.global_epochs}] - Global Training accuracy: {train_acc:.4f}, Test accuracy: {test_acc:.4f}, Best test accuracy: {max_acc:.4f}, Train elapse: {time.time() - global_start_time:.2f}s")

elif args.strategy == 'fedlec': # label alignment calibration
    nets, local_model_meta_data, layer_type = init_nets(args.n_parties, args)
    global_models, global_model_meta_data, global_layer_type = init_nets(1, args)
    global_model = global_models[0]

    global_param = global_model.state_dict()

    if not args.not_same_initial:
        for _, net in nets.items():
            net.load_state_dict(global_param)

    global_model.to(device)

    max_acc = -1.
    for epoch in range(args.global_epochs):
        global_start_time = time.time()
        arr = np.arange(args.n_parties)
        np.random.shuffle(arr)
        selected = arr[:int(args.n_parties * args.frac)]

        global_param = global_model.state_dict()

        for idx in selected:
            nets[idx].load_state_dict(global_param)

        # fedavg local training
        for net_id, net in nets.items():
            if net_id not in selected:
                continue

            net.to(device)
            train_ds_local = train_all_in_list[net_id]
            train_dl_local = torch.utils.data.DataLoader(dataset=train_ds_local, batch_size=args.batch_size, shuffle=True)
            logger.info(f'Training network {net_id}. n_training: {len(train_ds_local)}')

            if args.optimizer == 'adam':
                optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr, weight_decay=args.weight_decay)
            elif args.optimizer == 'sgd':
                optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
            else:
                raise NotImplementedError(f'Not support {args.optimizer}')
            criterion = nn.CrossEntropyLoss().to(device)

            num_per_class = [traindata_cls_counts[net_id][c] if c in traindata_cls_counts[net_id].keys() else 0 for c in range(args.num_classes)]
            num_per_class = torch.tensor(num_per_class).float().to(device)
            prior = num_per_class / num_per_class.sum()
            no_exist_label = torch.where(prior == 0)[0]
            exist_label = torch.where(prior != 0)[0]
            no_exist_label = no_exist_label.clone().detach().int().to(device)   # torch.tensor(no_exist_label)
            exist_label = exist_label.clone().detach().int().to(device)    # torch.tensor(exist_label).int().to(device)
            exist_prior = prior[exist_label]

            for l_epoch in range(args.local_epochs):
                local_start_time = time.time()
                epoch_loss_collector = []
                for x, target in train_dl_local:
                    x, target = x.to(device), target.to(device)

                    optimizer.zero_grad()
                    x.requires_grad = True
                    target.requires_grad = False
                    target = target.long()

                    # event-driven data, repeat num is 0
                    if repeat_num:
                        x = x.unsqueeze(0).repeat(args.T, 1, 1, 1, 1) # -> (T, B, C, H, W)
                    else: # dvs data
                        x = x.transpose(0, 1) # (B, T, C, H, W) -> (T, B, C, H, W)

                    out = net(x)
                
                    prior_celoss = criterion(out, target)
                    cls_weight = (num_per_class.float() / torch.sum(num_per_class.float())).to(device)
                    balanced_prior = torch.tensor(1. / args.num_classes).float().to(device)
                    pred_spread = (out - torch.log(prior + 1e-9) + torch.log(balanced_prior + 1e-9)).T * (target != torch.arange(0, args.num_classes).view(-1, 1).type_as(target)) # (cls, B)
                    batch_num = pred_spread.size(-1)
                    l_gc = prior_celoss - 0.005 * torch.sum((np.log(batch_num) - torch.logsumexp(pred_spread, -1)) * cls_weight)

                    teach_output = global_model(x).detach()
                    output_no_exist_log_soft = nn.functional.log_softmax(out[:, no_exist_label], dim=1)
                    output_no_exist_teacher_soft = nn.functional.softmax(teach_output[:, no_exist_label], dim=1)
                    kl = nn.KLDivLoss(reduction='batchmean')
                    l_ad = kl(output_no_exist_log_soft, output_no_exist_teacher_soft)

                    loss = l_gc + l_ad * args.lamda

                    loss.backward()
                    optimizer.step()

                    epoch_loss_collector.append(loss.item())

                    functional.reset_net(net)
                    functional.reset_net(global_model)

                epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
                if (1 + l_epoch) % 5 == 0:
                    logger.info(f'Local Epoch [{1 + l_epoch}/{args.local_epochs}] - Loss: {epoch_loss:.4f}, Time elapse: {time.time() - local_start_time:.2f}s')

            train_acc = compute_accuracy(net, train_dl_local, device, repeat_num)
            test_acc = compute_accuracy(net, test_dl_global, device, repeat_num)
            net.to('cpu')

            logger.info(f"net {net_id} Training accuracy: {train_acc:.4f}, Test accuracy: {test_acc:.4f}")

        # global updating
        total_data_points = sum([len(net_dataidx_map[r]) for r in selected])
        fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in selected]

        for idx in range(len(selected)):
            net_param = nets[selected[idx]].cpu().state_dict()
            # add Laplacian noise
            if args.eps > 0:
                for key in net_param:
                    net_param[key] = net_param[key].float() + Laplace(0, args.eps).expand(net_param[key].shape).sample()

            if idx == 0:
                for key in net_param:
                    global_param[key] = net_param[key] * fed_avg_freqs[idx]
            else:
                for key in net_param:
                    global_param[key] += net_param[key] * fed_avg_freqs[idx]

        global_model.load_state_dict(global_param)
        train_acc = compute_accuracy(global_model, train_dl_global, device, repeat_num)
        test_acc = compute_accuracy(global_model, test_dl_global, device, repeat_num)

        global_acc_record.append(test_acc)
        if max_acc <= test_acc:
            max_acc = test_acc

        logger.info(f"Epoch [{epoch + 1}/{args.global_epochs}] - Global Training accuracy: {train_acc:.4f}, Test accuracy: {test_acc:.4f}, Best test accuracy: {max_acc:.4f}, Train elapse: {time.time() - global_start_time:.2f}s")

else:
    raise NotImplementedError(f'Unknown federate strategy: {args.strategy}')

ensure_dir(args.result_dir)
act = 'snn' if args.snn else 'ann'
torch.save(global_param, f'./{args.result_dir}/saved_{act}_{args.model}_{args.dataset}_T{args.T}_{args.strategy}_{args.partition}_{args.noise_type}_noise{args.noise}_{args.global_epochs}GE_{args.local_epochs}LE.pt')

global_acc_dynamics = pd.DataFrame({
    'epoch': list(range(args.global_epochs)),
    'test_acc': global_acc_record
})
global_acc_dynamics.to_csv(f'./{args.result_dir}/acc_{act}_{args.model}_{args.dataset}_T{args.T}_{args.strategy}_{args.partition}_{args.noise_type}_noise{args.noise}_{args.global_epochs}GE_{args.local_epochs}LE_{args.n_parties}party_frac{args.frac}.csv', index=False)
