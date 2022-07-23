"""
Code from the following paper:
@inproceedings{wu2020skip,
    title={Skip connections matter: On the transferability of adversarial examples generated with resnets},
    author={Wu, Dongxian and Wang, Yisen and Xia, Shu-Tao and Bailey, James and Ma, Xingjun},
    booktitle={ICLR},
    year={2020}
}
https://github.com/csdongxian/skip-connections-matter
"""

import numpy as np
import torch
import torch.nn as nn


def backward_hook(gamma):
    # implement SGM through grad through ReLU
    def _backward_hook(module, grad_in, grad_out):
        if isinstance(module, nn.ReLU):
            return (gamma * grad_in[0],)
    return _backward_hook


def backward_hook_norm(module, grad_in, grad_out):
    # normalize the gradient to avoid gradient explosion or vanish
    grad_in_norm = []
    for i, grad_in_i in enumerate(grad_in):
        std = torch.std(grad_in_i)
        grad_in_norm.append(grad_in_i / std)
    return tuple(grad_in_norm)


def register_hook_for_resnet(model, arch, gamma):
    # There is only 1 ReLU in Conv module of ResNet-18/34
    # and 2 ReLU in Conv module ResNet-50/101/152
    if arch in ['resnet50', 'resnet101', 'resnet152']:
        gamma = np.power(gamma, 0.5)
    backward_hook_sgm = backward_hook(gamma)

    for name, module in model.named_modules():
        if 'relu' in name and not '0.relu' in name:
            # only to the last ReLU of each layer
            module.register_backward_hook(backward_hook_sgm)

        # e.g., 1.layer1.1, 1.layer4.2, ...
        # if len(name.split('.')) == 3:
        if len(name.split('.')) >= 2 and 'layer' in name.split('.')[-2]:
            module.register_backward_hook(backward_hook_norm)

def register_hook_for_preresnet(model, arch, gamma):
    # There are 3 ReLU in Conv module of PreResNet110
    if arch in ['PreResNet110']:
        gamma = np.power(gamma, 1/3)
    else:
        raise ValueError('Arch not supported')
    backward_hook_sgm = backward_hook(gamma)

    for name, module in model.named_modules():
        if 'relu' in name and 'layer' in name and '.0.relu' not in name:
            # .0.relu to skip the relu of downsampling ('.0.' is important to keep layerX.10.relu)
            # do not apply to last relu 'relu'
            module.register_backward_hook(backward_hook_sgm)

        # e.g., 1.layer1.1, 1.layer4.2, ...
        # if len(name.split('.')) == 3:
        if len(name.split('.')) >= 2 and 'layer' in name.split('.')[-2]:
            module.register_backward_hook(backward_hook_norm)


def register_hook_for_densenet(model, arch, gamma):
    # There are 2 ReLU in Conv module of DenseNet-121/169/201.
    gamma = np.power(gamma, 0.5)
    backward_hook_sgm = backward_hook(gamma)
    for name, module in model.named_modules():
        if 'relu' in name and not 'transition' in name:
            module.register_backward_hook(backward_hook_sgm)
