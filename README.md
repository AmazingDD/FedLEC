# FedLEC
Official code for "FedLEC: Effective Federated Learning Method Under Extreme Label Skews for Spiking Neural Networks" (AAAI2025)

## Overview
<p align="center">
<img src="./assets/FedLAC.png" align="center" width="70%" style="margin: 0 auto">
</p>

## Requirements

```
torch==2.0.1
torchvision==0.15.2
spikingjelly==0.0.0.0.14
numpy==1.23.5
pandas==1.5.3
scikit-learn==1.2.1
```

## How to run

```
python main.py -data_dir=. -dataset=cifar10 -model=vgg9 -strategy=fedlac -np=10 -frac=0.2 -gpu=0 -partition=noniid-c-dir -alpha=0.5 -snn -T=4 -desc=Example
```

The experimental options implemented in paper are listed below:
| Parameter Name | Support Options |
|----------|----------|
| partition | fedavg/fednova/fedprox/scaffold/fedlec |
| dataset | cifar10/svhn/nmnist/cifar10-dvs |
| model | vgg9/vit1/resnet10 |
| partition | noniid-cnum-2/noniid-c-dir |

The functionalities of other parameters can refer to the description text in `main.py`

## Cite

Please cite the following paper if you find our work contributes to yours in any way:

```

```