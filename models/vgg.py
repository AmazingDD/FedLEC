import re

import torch.nn as nn
from spikingjelly.activation_based import layer, functional, neuron, surrogate

from utils import INPUT_SIZE

vgg_cfg = {
    'VGG5' : [64, 'P', 128, 128, 'P'],
    'VGG9':  [64, 64, 'P', 128, 128, 'P', 256, 256, 256, 'P'],
    'VGG11': [64, 'P', 128, 'p', 256, 256, 'P', 512, 512, 'P', 512, 512, 'P'],
    'VGG13': [64, 64, 'P', 128, 128, 'P', 256, 256, 'P', 512, 512 , 'P', 512, 512, 'P'],
    'VGG16': [64, 64, 'P', 128, 128, 'P', 256, 256, 256, 'P', 512, 512, 512, 'P', 512, 512, 512, 'P'],
    'VGG19': [64, 64, 'P', 128, 128, 'P', 256, 256, 256, 256, 'P', 512, 512, 512, 512, 'P', 512, 512, 512, 512, 'P']
}

class SimpleCNN(nn.Module):
    def __init__(self, args):
        super(SimpleCNN, self).__init__()

        C, H, W = INPUT_SIZE[args.dataset]
        self.bias_flag = False if args.snn else True
        self.conv_fc = nn.Sequential(
            layer.Conv2d(C, 6, kernel_size=5),
            neuron.LIFNode(detach_reset=True, surrogate_function=surrogate.ATan()) if args.snn else nn.ReLU(),
            layer.MaxPool2d(2, 2),
            layer.Conv2d(6, 16, kernel_size=5),
            neuron.LIFNode(detach_reset=True, surrogate_function=surrogate.ATan()) if args.snn else nn.ReLU(),
            layer.MaxPool2d(2, 2),
            layer.Flatten(),
            layer.Linear((((H - 5 + 1) // 2 - 5 + 1) // 2) * (((W - 5 + 1) // 2 - 5 + 1) // 2) * 16, 120),
            neuron.LIFNode(detach_reset=True, surrogate_function=surrogate.ATan()) if args.snn else nn.ReLU(),
            layer.Linear(120, 84),
            neuron.LIFNode(detach_reset=True, surrogate_function=surrogate.ATan()) if args.snn else nn.ReLU(),
            layer.Linear(84, args.num_classes),
        )

        functional.set_step_mode(self, 'm')

    def forward(self, x):
        x = self.conv_fc(x)

        return x.mean(0)

class VGG(nn.Module):
    def __init__(self, args):
        ''' VGG '''
        super(VGG, self).__init__()

        C, H, W = INPUT_SIZE[args.dataset]
        self.bias_flag = False if args.snn else True
        pattern = re.compile(r'\d+')
        layer_num = int(pattern.findall(args.model)[0])
        assert int(layer_num) in [5, 9, 11, 13, 16, 19], f'current layer settings of {args.model} not support!'

        conv = []
        in_channel = C

        for x in vgg_cfg[f'VGG{layer_num}']:
            if x == 'P':
                conv.append(layer.MaxPool2d(kernel_size=2))
                H //= 2
                W //= 2
            else:
                out_channel = x
                conv.append(
                    layer.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=self.bias_flag))
                conv.append(layer.BatchNorm2d(out_channel))
                conv.append(
                    neuron.LIFNode(detach_reset=True, surrogate_function=surrogate.ATan()) if args.snn else nn.ReLU())
                in_channel = out_channel

        self.features = nn.Sequential(*conv)

        if int(layer_num) in [5, 9]:
            self.fc = nn.Sequential(
                layer.Flatten(),
                layer.Linear(in_channel * H * W, 1024, bias=self.bias_flag),
                neuron.LIFNode(detach_reset=True, surrogate_function=surrogate.ATan()) if args.snn else nn.ReLU(),
                layer.Linear(1024, args.num_classes, bias=self.bias_flag),
            )
        else:
            self.fc = nn.Sequential(
                layer.Flatten(),
                layer.Linear(in_channel * H * W, 4096, bias=self.bias_flag),
                neuron.LIFNode(detach_reset=True, surrogate_function=surrogate.ATan()) if args.snn else nn.ReLU(),
                layer.Linear(4096, 4096, bias=self.bias_flag),
                neuron.LIFNode(detach_reset=True, surrogate_function=surrogate.ATan()) if args.snn else nn.ReLU(),
                layer.Linear(4096, args.num_classes, bias=self.bias_flag),
            )

        # for m in self.modules():
        #     if isinstance(m, (layer.Conv2d, layer.Linear)):
        #         torch.nn.init.xavier_uniform_(m.weight, gain=2)

        functional.set_step_mode(self, 'm')

    def forward(self, x):
        x = self.features(x)  # (T, B, C, H, W)
        x = self.fc(x) # -> (T, B, num_cls)

        return x.mean(0) # -> (B, num_cls)
    
class VGGEncoder(nn.Module):
    def __init__(self, args):
        super(VGGEncoder, self).__init__()

        C, H, W = INPUT_SIZE[args.dataset]
        self.bias_flag = False if args.snn else True
        pattern = re.compile(r'\d+')
        layer_num = int(pattern.findall(args.model)[0])
        assert int(layer_num) in [5, 9, 11, 13, 16, 19], f'current layer settings of {args.model} not support!'

        conv = []
        in_channel = C

        for x in vgg_cfg[f'VGG{layer_num}']:
            if x == 'P':
                conv.append(layer.MaxPool2d(kernel_size=2))
                H //= 2
                W //= 2
            else:
                out_channel = x
                conv.append(
                    layer.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=self.bias_flag))
                conv.append(layer.BatchNorm2d(out_channel))
                conv.append(
                    neuron.LIFNode(detach_reset=True, surrogate_function=surrogate.ATan()) if args.snn else nn.ReLU())
                in_channel = out_channel

        self.conv_layer = nn.Sequential(*conv)

        if int(layer_num) in [5, 9]:
            self.fc_layer = nn.Sequential(
                layer.Flatten(),
                layer.Linear(in_channel * H * W, 1024, bias=self.bias_flag),
                neuron.LIFNode(detach_reset=True, surrogate_function=surrogate.ATan()) if args.snn else nn.ReLU(),
                # layer.Linear(1024, args.num_classes, bias=self.bias_flag),
            )
        else:
            self.fc_layer = nn.Sequential(
                layer.Flatten(),
                layer.Linear(in_channel * H * W, 4096, bias=self.bias_flag),
                neuron.LIFNode(detach_reset=True, surrogate_function=surrogate.ATan()) if args.snn else nn.ReLU(),
                layer.Linear(4096, 4096, bias=self.bias_flag),
                neuron.LIFNode(detach_reset=True, surrogate_function=surrogate.ATan()) if args.snn else nn.ReLU(),
                # layer.Linear(4096, args.num_classes, bias=self.bias_flag),
            )

        functional.set_step_mode(self, 'm')

    def forward(self, x):
        x = self.conv_layer(x)  # (T, B, C, H, W)
        x = self.fc_layer(x) # -> (T, B, num_cls)

        return x # -> (T, B, D)
    

class VGGClassifier(nn.Module):
    def __init__(self, args, num_k=None):
        super(VGGClassifier, self).__init__()

        self.bias_flag = False if args.snn else True
        pattern = re.compile(r'\d+')
        layer_num = int(pattern.findall(args.model)[0])
        assert int(layer_num) in [5, 9, 11, 13, 16, 19], f'current layer settings of {args.model} not support!'

        hidden_dim = 1024 if int(layer_num) in [5, 9] else 4096
        if num_k is not None:
            hidden_dim = hidden_dim * num_k
        self.fc = layer.Linear(hidden_dim, args.num_classes, bias=self.bias_flag)

        functional.set_step_mode(self, 'm')

    def forward(self, x):
        x = self.fc(x)
        return x.mean(0) # -> (B, D)
    