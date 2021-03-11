import torch
import torch.nn as nn
import einops
import torch.nn.functional as F
from RNN import RNN
class seq2seq(nn.Module):
    def __init__(self, hidden_size: int, embedding_size: int, 
                 num_layers: int, bidirectional: bool):
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.encoder = RNN(self.hidden_size, self.embedding_size, 
                           self.num_layers, self.bidirectional)
        self.decoder = RNN(self.hidden_size, self.embedding_size,
                           self.num_layers, self.bidirectional)
    
    def forward(self, )