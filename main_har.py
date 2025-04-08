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
parser.add_argument('-model', type=str, default='cnn', help='neural network used in training')
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
repeat_num = args.T

# reproducibility
seed = args.init_seed
np.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)

logger = Logger(args, args.desc)
logger.info(str(args))

logger.info("Loading data")
X_train, y_train, X_test, y_test, net_dataidx_map = load_data_har(args)

n_parties = len(net_dataidx_map)

traindata_cls_counts = {}
for net_i, dataidx in net_dataidx_map.items():
    unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
    tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
    traindata_cls_counts[net_i] = tmp
logger.info(f'Finish partitioning, data statistics: {str(traindata_cls_counts)}')

train_all_in_list = [] # used to store local train dataset
test_all_in_list = [] # used to store local test dataset

for party_id in range(n_parties):
    dataidxs = net_dataidx_map[party_id]

    train_dl_local, test_dl_local, train_ds_local, test_ds_local = get_dataloader(
            (X_train, X_test), 
            (y_train, y_test), 
            args.batch_size, 32, dataidxs, 0, data_name=args.dataset)
    
    train_all_in_list.append(train_ds_local)
    test_all_in_list.append(test_ds_local)

train_ds_global = torch.utils.data.ConcatDataset(train_all_in_list)
train_dl_global = torch.utils.data.DataLoader(dataset=train_ds_global, batch_size=args.batch_size, shuffle=True)
test_ds_global = torch.utils.data.ConcatDataset(test_all_in_list)
test_dl_global = torch.utils.data.DataLoader(dataset=test_ds_global, batch_size=128, shuffle=False)
logger.info(f'length of train_dl_global: {len(train_ds_global)}')

logger.info('Inint models')
def init_nets(n_parties, args):
    nets = {net_i: None for net_i in range(n_parties)}

    for net_i in range(n_parties):
        net = SimpleCNN(args)

        nets[net_i] = net

    model_meta_data = []
    layer_type = []
    for k, v in nets[0].state_dict().items():
        model_meta_data.append(v.shape)
        layer_type.append(k)

    return nets, model_meta_data, layer_type

global_acc_record = []

if args.strategy == 'fedavg':
    nets, local_model_meta_data, layer_type = init_nets(n_parties, args)
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
        arr = np.arange(n_parties)
        np.random.shuffle(arr)
        selected = arr[:int(n_parties * args.frac)]

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
                    x, target = x.float().to(device), target.to(device)

                    optimizer.zero_grad()
                    x.requires_grad = True
                    target.requires_grad = False
                    target = target.long()

                    x = x.unsqueeze(0).repeat(args.T, 1, 1, 1, 1) # -> (T, B, C, H, W)

                    out = net(x)

                    loss = criterion(out, target)

                    loss.backward()
                    optimizer.step()

                    epoch_loss_collector.append(loss.item())

                    functional.reset_net(net)

                epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
                if (1 + l_epoch) % 5 == 0:
                    logger.info(f'Local Epoch [{1 + l_epoch}/{args.local_epochs}] - Loss: {epoch_loss:.4f}, Time elapse: {time.time() - local_start_time:.2f}s')
            net.to('cpu')
            logger.info(f'Finish net {net_id} Training')

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
        test_acc = compute_accuracy(global_model, test_dl_global, device, repeat_num)

        global_acc_record.append(test_acc)
        if max_acc <= test_acc:
            max_acc = test_acc

        logger.info(f"Epoch [{epoch + 1}/{args.global_epochs}] - Test accuracy: {test_acc:.4f}, Best test accuracy: {max_acc:.4f}, Train elapse: {time.time() - global_start_time:.2f}s")

elif args.strategy == 'fedlec': 
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
        selected = arr[:int(n_parties * args.frac)]

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
                    x, target = x.float().to(device), target.to(device)

                    optimizer.zero_grad()
                    x.requires_grad = True
                    target.requires_grad = False
                    target = target.long()

                    x = x.unsqueeze(0).repeat(args.T, 1, 1, 1, 1)

                    out = net(x)

                    logits = out + torch.log(prior + 1e-9)
                    prior_celoss = criterion(logits, target)

                    cls_weight = (num_per_class.float() / torch.sum(num_per_class.float())).to(device)
                    balanced_prior = torch.tensor(1. / args.num_classes).float().to(device)
                    pred_spread = (out - torch.log(prior + 1e-9) + torch.log(balanced_prior + 1e-9)).T * (target != torch.arange(0, args.num_classes).view(-1, 1).type_as(target)) # (cls, B)
                    batch_num = pred_spread.size(-1)
                    l_gc = -torch.sum((np.log(batch_num) - torch.logsumexp(pred_spread, -1)) * cls_weight)

                    teach_output = global_model(x).detach()
                    output_no_exist_log_soft = nn.functional.log_softmax(out[:, no_exist_label], dim=1)
                    output_no_exist_teacher_soft = nn.functional.softmax(teach_output[:, no_exist_label], dim=1)
                    kl = nn.KLDivLoss(reduction='batchmean')
                    l_ad = kl(output_no_exist_log_soft, output_no_exist_teacher_soft)

                    loss = prior_celoss + 0.005 * l_gc + l_ad * args.lamda # \theta=0.005

                    loss.backward()
                    optimizer.step()

                    epoch_loss_collector.append(loss.item())

                    functional.reset_net(net)
                    functional.reset_net(global_model)

                epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
                if (1 + l_epoch) % 5 == 0:
                    logger.info(f'Local Epoch [{1 + l_epoch}/{args.local_epochs}] - Loss: {epoch_loss:.4f}, Time elapse: {time.time() - local_start_time:.2f}s')

            net.to('cpu')
            logger.info(f"Finish net {net_id} training")

        # global updating
        total_data_points = sum([len(net_dataidx_map[r]) for r in selected])
        fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in selected]

        for idx in range(len(selected)):
            net_param = nets[selected[idx]].cpu().state_dict()
            if args.eps > 0:
                from torch.distributions.laplace import Laplace
                for key in net_param:
                    net_param[key] = net_param[key].float() +  Laplace(0, args.eps).expand(net_param[key].shape).sample()

            if idx == 0:
                for key in net_param:
                    global_param[key] = net_param[key] * fed_avg_freqs[idx]
            else:
                for key in net_param:
                    global_param[key] += net_param[key] * fed_avg_freqs[idx]

        global_model.load_state_dict(global_param)
        # train_acc = compute_accuracy(global_model, train_dl_global, device, repeat_num)
        test_acc = compute_accuracy(global_model, test_dl_global, device, repeat_num)

        global_acc_record.append(test_acc)
        if max_acc <= test_acc:
            max_acc = test_acc

        logger.info(f"Epoch [{epoch + 1}/{args.global_epochs}] - Test accuracy: {test_acc:.4f}, Best test accuracy: {max_acc:.4f}, Train elapse: {time.time() - global_start_time:.2f}s")

ensure_dir(args.result_dir)
act = 'snn' if args.snn else 'ann'

torch.save(global_param, f'./{args.result_dir}/saved_{act}_{args.model}_{args.dataset}_T{args.T}_{args.strategy}_{args.partition}_{args.noise_type}_noise{args.noise}_{args.global_epochs}GE_{args.local_epochs}LE.pt')

global_acc_dynamics = pd.DataFrame({
    'epoch': list(range(args.global_epochs)),
    'test_acc': global_acc_record
})
global_acc_dynamics.to_csv(f'./{args.result_dir}/{args.desc}_acc_{act}_{args.model}_{args.dataset}_T{args.T}_{args.strategy}_{args.partition}_{args.noise_type}_noise{args.noise}_{args.global_epochs}GE_{args.local_epochs}LE_{n_parties}party_frac{args.frac}.csv', index=False)