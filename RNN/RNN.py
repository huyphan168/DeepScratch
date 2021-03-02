import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class RNNcell(nn.Module):
    def __init__(self, hidden_size: int, embedding_size: int, activation: str='Tanh'):
        super(RNNcell, self).__init__()
        if activation == "ReLU":
            self.activation = nn.ReLU()
        elif activation == "Tanh":
            self.activation = nn.Tanh()
        else:
            self.activation = nn.Sigmoid()
        self.Waa = nn.Parameter(torch.zeros(hidden_size, hidden_size))
        self.Wax = nn.Parameter(torch.zeros(hidden_size, embedding_size))
        self.Ba = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, a, x):
        return self.activation(self.Waa*a + self.Wax*x + self.Ba)

class RNN(nn.Module):
    def __init__(self, hidden_size: int, embedding_size: int, num_layers: int, bidirectional: bool):
        super(RNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.bidirectional = bidirectional  
        if self.bidirectional:
            input_layer = nn.ModuleList([RNNcell(self.hidden_size, self.embedding_size) for i in range(1)])
            hidden_layer = nn.ModuleList([RNNcell(self.hidden_size, 2*self.hidden_size) for i in range(1)])
            self.mod = nn.ModuleList([input_layer])
            if self.num_layers > 1:
                self.mod.extend([hidden_layer for i in range(num_layers-1)])
        else:
            input_layer = RNNcell(self.hidden_size, self.embedding_size)
            hidden_layer = RNNcell(self.hidden_size, self.hidden_size)
            self.mod = nn.ModuleList([input_layer])
            if self.num_layers > 1:
                self.mod.extend([hidden_layer for i in range(num_layers-1)])

        

    def __call__(self, input_seq):
        seq_len = input_seq.size()[0]
        if self.bidirectional is False:
            a = [torch.zeros(self.hidden_size) for i in range(self.num_layers)]
            for t in range(seq_len):
                x = input_seq[t]
                for layer in range(self.num_layers):
                    x = self.mod[layer](a[layer], x)
                    a[layer] = x
            return a[-1]
        else:
            a = [torch.zeros(self.hidden_size) for i in range(self.num_layers*2)]
            h = [torch.zeros(self.hidden_size*2) for i in range(self.num_layers)]
            for layer in range(self.num_layers):
                for t in range(seq_len):
                    x = input_seq[t]
                    x_1 = self.mod[layer][0](a[layer*2], x)
                    a[layer*2] = x_1
                for t in reversed(range(seq_len)):
                    x = input_seq[t]
                    x_2 = self.mod[layer][1](a[layer*2], x)
                    a[layer*2 + 1] = x_2
                h = [torch.cat([a[i], a[i+1]]) for i in range(0,self.num_layers*2, 2)]
                input_seq = h
            return h[-1]
def _test():
    RNN_test = RNN(128, 60, 2, False)
    sequence = torch.rand(12,60)
    print(sequence.size())
    RNN_test(sequence)

if __name__ == "__main__":
    _test()