# YOLOv7 utils module
# Simplified utils module

import torch
import torch.nn as nn

class ModelEMA:
    """Model Exponential Moving Average"""
    def __init__(self, model, decay=0.9999):
        self.ema = model
        self.decay = decay
    
    def update(self, model):
        pass

# Export classes
__all__ = ['ModelEMA']
