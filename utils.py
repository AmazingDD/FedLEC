import os
import math
import datetime
import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from spikingjelly.activation_based import functional
from spikingjelly.datasets.n_mnist import NMNIST
from spikingjelly.datasets.cifar10_dvs import CIFAR10DVS
from spikingjelly.datasets.n_caltech101 import NCaltech101
from spikingjelly.datasets.dvs128_gesture import DVS128Gesture
from spikingjelly.datasets import split_to_train_test_set

INPUT_SIZE = {
    'mnist': (1, 28, 28),
    'nmnist': (2, 34, 34),
    'cifar10': (3, 32, 32),
    'cifar100': (3, 32, 32),
    'svhn': (3, 32, 32),
    'tinyimagenet': (3, 64, 64),
    'cifar10-dvs': (2, 128, 128),
    'dvs128gesture': (2, 128, 128),
    'ncaltech101': (2, 180, 240),
    'nmnist': (2, 34, 34), 
    'fmnist': (1, 28, 28)
}

CLASS_NUM = {
    'cifar10': 10,
    'cifar100': 100,
    'mnist': 10,
    'svhn': 10,
    'tinyimagenet': 200, 
    'cifar10-dvs': 10,
    'dvs128gesture': 11, # T=16
    'ncaltech101': 101,
    'nmnist': 10,
    'fmnist': 10,
}

NORM_PARAM = {
    'cifar10': [(0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)],
    'cifar100': [(0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)],
    'mnist': [(0.1307,), (0.3081,)],
    'tinyimagenet': [(0.5, 0.5, 0.5), (0.5, 0.5, 0.5)],
    'fmnist': [(0.1307,), (0.3081,)],
}

def ensure_dir(dirpath):
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)

def get_local_time():
    cur = datetime.datetime.now()
    cur = cur.strftime('%b-%d-%Y_%H-%M-%S')

    return cur

class Logger:
    ''' spikingjelly induce logging not work '''
    def __init__(self, args, desc='fl'):
        log_root = args.log_dir
        dir_name = os.path.dirname(log_root)
        ensure_dir(dir_name)

        act = 'snn' if args.snn else 'ann'

        out_dir = os.path.join(log_root, f'{args.dataset}_{args.strategy}_{args.model}_{act}_T{args.T}_b{args.batch_size}_lr{args.lr}_{args.global_epochs}GE_{args.local_epochs}LE_{args.n_parties}party_frac{args.frac}')
        ensure_dir(out_dir)

        logfilename = f'{desc}_{args.partition}_{args.noise_type}_noise{args.noise}_record.log' # _{get_local_time()}
        logfilepath = os.path.join(out_dir, logfilename)

        self.filename = logfilepath

        f = open(logfilepath, 'w', encoding='utf-8')
        f.write(str(args) + '\n')
        f.flush()
        f.close()


    def info(self, s=None):
        print(s)
        f = open(self.filename, 'a', encoding='utf-8')
        f.write(f'[{get_local_time()}] - {s}\n')
        f.flush()
        f.close()

def load_data(args):
    dataset = args.dataset

    if dataset == 'mnist':
        transform = transforms.Compose([transforms.ToTensor()])
        train_ds = torchvision.datasets.MNIST(
            root=args.data_dir, train=True, transform=transform, download=True)
        test_ds = torchvision.datasets.MNIST(
            root=args.data_dir, train=False, transform=transform, download=True)
        X_train, y_train = train_ds.data, train_ds.targets
        X_test, y_test = test_ds.data, test_ds.targets

    elif dataset == 'fmnist':
        transform = transforms.Compose([transforms.ToTensor()])
        train_ds = torchvision.datasets.FashionMNIST(
            root=args.data_dir, train=True, transform=transform, download=True)
        test_ds = torchvision.datasets.FashionMNIST(
            root=args.data_dir, train=False, transform=transform, download=True)
        X_train, y_train = train_ds.data, train_ds.targets
        X_test, y_test = test_ds.data, test_ds.targets

        X_train = X_train.unsqueeze(-1).numpy()
        X_test = X_test.unsqueeze(-1).numpy()
        y_test = y_test.numpy()
        y_train = y_train.numpy()

    elif dataset == 'cifar10':
        transform = transforms.Compose([transforms.ToTensor()])
        train_ds =  torchvision.datasets.CIFAR10(
            root=args.data_dir, train=True, download=True)
        test_ds =  torchvision.datasets.CIFAR10(
            root=args.data_dir, train=False, download=True)

        X_train, y_train = train_ds.data, train_ds.targets
        X_test, y_test = test_ds.data, test_ds.targets
    
    elif dataset == 'cifar100':
        transform = transforms.Compose([transforms.ToTensor()])
        train_ds =  torchvision.datasets.CIFAR100(
            root=args.data_dir, train=True, download=True)
        test_ds =  torchvision.datasets.CIFAR100(
            root=args.data_dir, train=False, download=True)

        X_train, y_train = train_ds.data, train_ds.targets
        X_test, y_test = test_ds.data, test_ds.targets

    elif dataset == 'svhn':
        transform = transforms.Compose([transforms.ToTensor()])
        train_ds =  torchvision.datasets.SVHN(
            root=args.data_dir, split='train', download=True)
        test_ds =  torchvision.datasets.SVHN(
            root=args.data_dir, split='test', download=True)

        X_train, y_train = train_ds.data, train_ds.labels
        X_test, y_test = test_ds.data, test_ds.labels

        B, C, H, W = X_train.shape
        X_train = X_train.reshape(B, H, W, C)
        B, C, H, W = X_test.shape
        X_test = X_test.reshape(B, H, W, C)

    elif dataset == 'tinyimagenet':
        transform = transforms.Compose([transforms.ToTensor()])
        train_ds = torchvision.datasets.ImageFolder(args.data_dir+'./train/', transform=transform)
        test_ds = torchvision.datasets.ImageFolder(args.data_dir+'./val/', transform=transform)

        X_train, y_train = np.array([s[0] for s in train_ds.samples]), np.array([int(s[1]) for s in train_ds.samples])
        X_test, y_test = np.array([s[0] for s in test_ds.samples]), np.array([int(s[1]) for s in test_ds.samples])

        X_train = np.array([train_ds.transform(train_ds.loader(p)).numpy() for p in X_train])
        X_test = np.array([test_ds.transform(test_ds.loader(p)).numpy() for p in X_test])

        B, C, H, W = X_train.shape
        X_train = X_train.reshape(B, H, W, C)
        B, C, H, W = X_test.shape
        X_test = X_test.reshape(B, H, W, C)

    elif dataset == 'cifar10-dvs':
        ds = CIFAR10DVS(
                root=args.data_dir, 
                data_type='frame', 
                frames_number=args.T, 
                split_by='number')
        train_dataset, test_dataset = split_to_train_test_set(0.8, ds, args.num_classes)

        X_train = np.array([xy[0] for xy in train_dataset])
        y_train = np.array([xy[1] for xy in train_dataset])
        X_test = np.array([xy[0] for xy in test_dataset])
        y_test = np.array([xy[1] for xy in test_dataset])

        B, T, C, H, W = X_train.shape
        X_train = X_train.reshape(B, T, H, W, C)
        B, T, C, H, W = X_test.shape
        X_test = X_test.reshape(B, T, H, W, C)

    elif dataset == 'nmnist':
        train_dataset = NMNIST(root=args.data_dir, train=True, data_type='frame', frames_number=args.T, split_by='number')
        test_dataset = NMNIST(root=args.data_dir, train=False, data_type='frame', frames_number=args.T, split_by='number')

        X_train = np.array([xy[0] for xy in train_dataset])
        y_train = np.array([xy[1] for xy in train_dataset])
        X_test = np.array([xy[0] for xy in test_dataset])
        y_test = np.array([xy[1] for xy in test_dataset])

        B, T, C, H, W = X_train.shape
        X_train = X_train.reshape(B, T, H, W, C)
        B, T, C, H, W = X_test.shape
        X_test = X_test.reshape(B, T, H, W, C)
        
    elif dataset == 'ncaltech101':
        ds = NCaltech101(
            root=args.data_dir, 
            data_type='frame', 
            frames_number=args.T, 
            split_by='number')
        
        train_dataset, test_dataset = split_to_train_test_set(0.8, ds, args.num_classes)

        X_train = np.array([xy[0] for xy in train_dataset])
        y_train = np.array([xy[1] for xy in train_dataset])
        X_test = np.array([xy[0] for xy in test_dataset])
        y_test = np.array([xy[1] for xy in test_dataset])

        B, T, C, H, W = X_train.shape
        X_train = X_train.reshape(B, T, H, W, C)
        B, T, C, H, W = X_test.shape
        X_test = X_test.reshape(B, T, H, W, C)

    elif dataset == 'dvs128gesture':
        train_dataset = DVS128Gesture(root=args.data_dir, train=True, data_type='frame', frames_number=args.T, split_by='number')
        test_dataset = DVS128Gesture(root=args.data_dir, train=False, data_type='frame', frames_number=args.T, split_by='number')

        X_train = np.array([xy[0] for xy in train_dataset])
        y_train = np.array([xy[1] for xy in train_dataset])
        X_test = np.array([xy[0] for xy in test_dataset])
        y_test = np.array([xy[1] for xy in test_dataset])

        B, T, C, H, W = X_train.shape
        X_train = X_train.reshape(B, T, H, W, C)
        B, T, C, H, W = X_test.shape
        X_test = X_test.reshape(B, T, H, W, C)

    else:
        raise NotImplementedError(f'Invalid dataset name {dataset}')
    
    y_train, y_test = np.array(y_train), np.array(y_test)

    return X_train, y_train, X_test, y_test

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1., net_id=None, total=0):
        self.std = std
        self.mean = mean
        self.net_id = net_id
        self.num = int(math.sqrt(total))
        if self.num * self.num < total:
            self.num += 1

    def __call__(self, tensor):
        if self.net_id is None:
            return tensor + torch.randn(tensor.size()) * self.std + self.mean
        else:
            tmp = torch.randn(tensor.size())
            filt = torch.zeros(tensor.size())
            size = int(28 / self.num)
            row = int(self.net_id / size)
            col = self.net_id % size
            for i in range(size):
                for j in range(size):
                    filt[..., row * size + i, col * size + j] = 1
            tmp = tmp * filt
            return tensor + tmp * self.std + self.mean
        
    def __repr__(self) -> str:
        return self.__class__.__name__ + f'(mean={self.mean}, std={self.std})'

def get_dataloader(Xs, ys, train_batch_size, test_batch_size=32, dataidxs=None, noise_level=0, net_id=None, total=0, data_name=None):
    class CustomDataset(Dataset):
        def __init__(self, X, y, dataidxs=None, transform=None) -> None:
            super().__init__()

            if dataidxs is None:
                self.data = X
                self.target = y
            else:
                self.data = X[dataidxs]
                self.target = y[dataidxs]

            self.transform = transform

        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, index):
            img, target = self.data[index], self.target[index]

            if self.transform is not None:
                if len(img.shape) <= 3: # common 2d/3d images
                    img = self.transform(img)
                else: # dvs-images with additional time dimension T
                    img = torch.stack([self.transform(i) for i in img], dim=0) 

            return img, target
        
    X_train, X_test = Xs
    y_train, y_test = ys

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(NORM_PARAM[data_name][0], NORM_PARAM[data_name][1]) if data_name in NORM_PARAM.keys() else transforms.Lambda(lambda x: x),
        AddGaussianNoise(0., noise_level, net_id, total)])
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(NORM_PARAM[data_name][0], NORM_PARAM[data_name][1]) if data_name in NORM_PARAM.keys() else transforms.Lambda(lambda x: x),
        AddGaussianNoise(0., noise_level, net_id, total)])

    train_ds = CustomDataset(X_train, y_train, dataidxs, transform=transform_train)
    test_ds = CustomDataset(X_test, y_test, transform=transform_test)

    train_dl = DataLoader(train_ds, batch_size=train_batch_size, shuffle=True, drop_last=False)
    test_dl = DataLoader(test_ds, batch_size=test_batch_size, shuffle=False, drop_last=False)

    return train_dl, test_dl, train_ds, test_ds

def compute_accuracy(model, dataloader, device='cpu', repeat_num=4):
    was_training = False
    if model.training:
        model.eval()
        was_training = True
    
    correct, total = 0, 0
    with torch.no_grad():
        for x, target in dataloader:
            x, target = x.to(device), target.to(device)

            if repeat_num:
                x = x.unsqueeze(0).repeat(repeat_num, 1, 1, 1, 1) # always (T, B, C, H, W)
            else:
                x = x.transpose(0, 1)
            out = model(x)
            _, pred_label = torch.max(out.data, 1)

            total += x.data.size()[1] # add B
            correct += (pred_label == target.data).sum().item()

            functional.reset_net(model)

    if was_training:
        model.train()

    return correct / float(total)
