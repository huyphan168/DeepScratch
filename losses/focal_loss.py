import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class FocalLoss(nn.Module):
    def __init__(self, gamma: float, alpha: torch.tensor, use_sigmoid: bool=True):
        self.gamma = gamma 
        self.alpha = alpha
        self.use_sigmoid = use_sigmoid
    
    def forward(self, logits,y):
        p = F.sigmoid(logits)
        fomulate_coef = -((1-p)*self.alpha)**self.gamma 
        focal_vector = fomulate_coef*torch.exp(p)*y
        return focal_vector.mean()