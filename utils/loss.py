#!/usr/bin/env python3
"""
YOLO loss utilities
"""

import torch
import torch.nn as nn

class SigmoidBin(nn.Module):
    """Sigmoid binary classification"""
    def __init__(self, bin_count=10, min=0.0, max=1.0, reg_scale=2.0, use_loss_regression=True, use_fw_regression=True, BCE_weight=1.0, smooth_eps=0.0):
        super(SigmoidBin, self).__init__()
        self.bin_count = bin_count
        self.length = bin_count + 1
        self.min = min
        self.max = max
        self.scale = float(max - min)
        self.shift = self.scale / 2.0

        self.use_loss_regression = use_loss_regression
        self.use_fw_regression = use_fw_regression
        self.reg_scale = reg_scale
        self.BCE_weight = BCE_weight

        start = min + (self.scale/2.0) / self.bin_count
        end = max - (self.scale/2.0) / self.bin_count
        step = self.scale / self.bin_count
        self.step = step
        #print(f" start = {start}, end = {end}, step = {step} ")

        bins = torch.range(start, end + 0.0001, step).float()
        self.register_buffer('bins', bins)

        self.cp = 1.0 - 0.5 * smooth_eps
        self.cn = 0.5 * smooth_eps

        self.BCEbins = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([BCE_weight]))
        self.MSELoss = nn.MSELoss()

    def get_length(self):
        return self.length

    def forward(self, pred, target, mask):
        # B, C, W, H
        assert pred.shape[0] == target.shape[0]
        assert pred.shape[1] == self.length
        assert pred.shape[2] == pred.shape[3]  # W = H
        assert pred.shape[2] == target.shape[2]
        assert pred.shape[3] == target.shape[3]
        B, C, W, H = pred.shape
        # mask = (mask > 1e-8).float()
        per_elements = self.bin_count
        pre = pred.permute(0, 2, 3, 1).contiguous().view(B, W, H, C, 1)
        pre = pre.view(B, W, H, per_elements, 2)
        pre = pre.view(B, W, H, per_elements, 2)

        M = mask.permute(0, 2, 3, 1).contiguous().view(B, W, H, 1)
        M = M.float()

        target = target.permute(0, 2, 3, 1).contiguous().view(B, W, H, 1)
        target = target.clamp(min=self.min, max=self.max)
        target = target.view(B, W, H, 1)
        target = target.repeat(1, 1, 1, per_elements)
        target = target.view(B, W, H, per_elements, 1)

        diff = target - self.bins
        diff = diff.abs()
        diff = diff.view(B, W, H, per_elements, 1)
        diff = diff.permute(0, 1, 2, 4, 3).contiguous()
        diff = diff.view(B, W, H, 1, per_elements)

        # CB focal
        smooth_eps = 0.0
        self.cp = 1.0 - 0.5 * smooth_eps
        self.cn = 0.5 * smooth_eps

        # focal loss
        alpha = 0.25
        gamma = 1.5
        # alpha=0.25
        # gamma=1.5
        dist = torch.abs(diff)
        dist = dist.view(B, W, H, per_elements)
        # dist = 1 - dist - self.cn
        dist = torch.clamp(dist, min=0.0, max=1.0)
        dist = dist.view(B, W, H, per_elements)
        # dist = 1.0 - dist
        # dist = dist ** gamma

        if self.use_fw_regression:
            loss = (1.0 - dist) * self.BCEbins(pre, target) + dist * self.MSELoss(pre, target)
        else:
            loss = self.BCEbins(pre, target)

        loss = loss * M
        loss = loss.sum() / (M.sum() + 1e-6)

        return loss

