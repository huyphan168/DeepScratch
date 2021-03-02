import torch
import torch.nn as nn
import torch.nn.functional as F

class KLLoss(nn.Module):
    def __init__(self):
        super.__init__()
    def forward(self, X, y):
        probs = F.softmax(X, dim=1)
        KL_vector = probs*torch.exp(probs*y)
        return KL_vector.mean()
