#!/usr/bin/env python3
"""
YOLO torch utilities
"""

import time
import math
import torch
import numpy as np

def time_synchronized():
    """Pytorch-accurate time"""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()

def fuse_conv_and_bn(conv, bn):
    """Fuse conv and bn layers"""
    # This is a simplified version - in practice you might want to implement actual fusion logic
    return conv

def model_info(model, verbose=False, img_size=640):
    """Model information"""
    n_p = sum(x.numel() for x in model.parameters())  # number parameters
    n_g = sum(x.numel() for x in model.parameters() if x.requires_grad)  # number gradients
    if verbose:
        print(f"{'layer':>5} {'name':>40} {'gradient':>9} {'parameters':>12} {'shape':>20} {'mu':>10} {'sigma':>10}")
        for i, (name, p) in enumerate(model.named_parameters()):
            name = name.replace('module_list.', '')
            print(f'{i:5g} {name:>40} {p.requires_grad:>9} {p.numel():>12} {str(list(p.shape)):>20} {p.mean():>10.3g} {p.std():>10.3g}')
    print(f"Model Summary: {len(list(model.modules()))} layers, {n_p} parameters, {n_g} gradients")

def scale_img(img, ratio=1.0, same_shape=False, gs=32):
    """Scale image"""
    if ratio == 1.0:
        return img
    h, w = img.shape[2:]
    s = (int(h * ratio), int(w * ratio))  # new size
    img = torch.nn.functional.interpolate(img, size=s, mode='bilinear', align_corners=False)  # resize
    if not same_shape:  # pad/crop img
        h, w = [math.ceil(x * ratio / gs) * gs for x in (h, w)]
    return torch.nn.functional.pad(img, [0, w - s[1], 0, h - s[0]], value=0.447)  # value = imagenet mean

def initialize_weights(model):
    """Initialize model weights"""
    for m in model.modules():
        t = type(m)
        if t is torch.nn.Conv2d:
            pass  # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif t is torch.nn.BatchNorm2d:
            m.eps = 1e-3
            m.momentum = 0.03
        elif t in [torch.nn.LeakyReLU, torch.nn.ReLU, torch.nn.ReLU6]:
            m.inplace = True

def select_device(device='', batch_size=None):
    """Select device for inference"""
    if device == 'cpu':
        return torch.device('cpu')
    elif device == 'cuda':
        return torch.device('cuda')
    else:
        # Auto-select device
        if torch.cuda.is_available():
            return torch.device('cuda')
        else:
            return torch.device('cpu')

def copy_attr(a, b, include=(), exclude=()):
    """Copy attributes from b to a"""
    for k, v in b.__dict__.items():
        if (len(include) and k not in include) or k.startswith('_') or k in exclude:
            continue
        else:
            setattr(a, k, v)
