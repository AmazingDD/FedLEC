import re
import copy
import torch
import torch.nn as nn
from spikingjelly.activation_based import layer, functional, neuron, surrogate

from utils import INPUT_SIZE

class Block(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, is_snn=True):
        super(Block, self).__init__()

        self.residual_function = nn.Sequential(
            neuron.LIFNode(tau=2.0, detach_reset=True, surrogate_function=surrogate.ATan()) if is_snn else nn.ReLU(),
            layer.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            layer.BatchNorm2d(out_channels),
            neuron.LIFNode(tau=2.0, detach_reset=True, surrogate_function=surrogate.ATan()) if is_snn else nn.ReLU(),
            layer.Conv2d(out_channels, out_channels * self.expansion, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(out_channels * self.expansion)
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                layer.MaxPool2d(2, 2),
                layer.Conv2d(in_channels, out_channels * self.expansion, kernel_size=1, stride=1, bias=False),
                layer.BatchNorm2d(out_channels * self.expansion),
            )

    def forward(self, x):
        return self.residual_function(x) + self.shortcut(x)


class Resnet(nn.Module):
    def __init__(self, args):
        super(Resnet, self).__init__()
        k = 1
        self.in_channels = 64 * k

        self.num_classes = args.num_classes

        pattern = re.compile(r'\d+')
        depths = int(pattern.findall(args.model)[0])
        if depths == 18:
            num_block = [2, 2, 2, 2]
        elif depths == 14:
            num_block = [1,1,2,2]
        elif depths == 22:
            num_block = [2,2,3,3]
        elif depths == 26:
            num_block = [3,3,3,3]
        elif depths == 10:
            num_block = [1, 1, 1, 1]
        elif depths == 34:
            num_block = [3, 4, 6, 3]
        else:
            raise NotImplementedError(f'Invalid model {args.model}, only support `resnet10`, `resnet18` and `resnet34`')

        C, H, W = INPUT_SIZE[args.dataset]

        self.conv1 = nn.Sequential(
            layer.Conv2d(C, self.in_channels, kernel_size=7, padding=3, stride=2),
            layer.BatchNorm2d(self.in_channels),
        )
        # conv1.weight
        self.block = Block
        self.layer1 = self._make_layer(self.block, 64 * k, num_block[0], 2, args)
        self.layer2 = self._make_layer(self.block, 128 * k, num_block[1], 2, args)
        self.layer3 = self._make_layer(self.block, 256 * k, num_block[2], 2, args)
        self.layer4 = self._make_layer(self.block, 512 * k, num_block[3], 2, args)
        self.lif = neuron.LIFNode(tau=2., v_threshold=0.5, detach_reset=True, surrogate_function=surrogate.ATan()) if args.snn else nn.ReLU()

        self.pool = layer.AdaptiveAvgPool2d((1, 1))

        self.fc = layer.Linear(512 * Block.expansion * k, self.num_classes)

        functional.set_step_mode(self, 'm')

    def _make_layer(self, block, out_channels, num_blocks, stride, args):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            if args.has_rate:
                layers.append(block(self.in_channels, out_channels, s, args.snn,args.rate))
            else:
                layers.append(block(self.in_channels, out_channels, s, args.snn))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.conv1(x)
        output = self.layer1(output)
        output = self.layer2(output)
        output = self.layer3(output)
        output = self.layer4(output)
        output = self.lif(output)

        output = self.pool(output)
        output = torch.flatten(output, 2)
        output = self.fc(output)

        return output.mean(0)  # B, n_cls