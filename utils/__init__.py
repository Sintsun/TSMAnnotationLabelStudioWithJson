# YOLOv7 utils module
# 簡化的 utils 模組

import torch
import torch.nn as nn

class ModelEMA:
    """模型指數移動平均"""
    def __init__(self, model, decay=0.9999):
        self.ema = model
        self.decay = decay
    
    def update(self, model):
        pass

# 導出類別
__all__ = ['ModelEMA']
