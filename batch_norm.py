import torch
import torch.nn as nn

class BatchNorm(nn.Module):
    def __init__(self, channels: int, eps: float=1e-5, affine: bool=True,
                 momentum: float=0.1, ema: bool=True):

        super.__init__()
        self.channels = channels
        self.eps = eps
        self.momentum = momentum
        self.ema = ema
        if self.ema:
            self.register_buffer('ema_mean', torch.zeros(self.channels))
            self.register_buffer('ema_var', torch.ones(self.channels))
        if self.affine:
            self.scale = nn.parameter(torch.ones(self.channels))
            self.shift = nn.parameter(torch.zeros(self.channels)) 
    
    def forward(self, is_test: bool, X: torch.tensor):
        x_shape = X.shape
        batch_size = x_shape[0]
        channels = x_shape[1]
        assert  channels == self.channels
        X = X.view(batch_size, channels, -1)
        if self.training and self.ema:
            mean = X.mean(axis=[0,2])
            mean_square = (X**2).mean(axis=[0,2])
            var = mean_square - mean**2
            if self.ema:
                self.ema_mean = self.momentum*mean + (1-self.momentum)*self.ema_mean
                self.ema_var = self.momentum*var + (1-self.momentum)*self.ema_var
        else:
            mean = self.ema_mean
            var = self.ema_var
        X_norm = (X-mean.view(1,-1,1)) / (torch.sqrt(var.view(1,-1,1)) + self.eps)
        if self.affine:
            X_norm = self.scale.view(1,-1,1)*X_norm + self.shift.view(1,-1,1)*X_norm
        X_norm = X_norm.view(x_shape)
        return X_norm