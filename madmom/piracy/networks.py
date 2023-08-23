"""
Code to handle the actual training/evaluation of neural nets with more data/torch
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm


# ====================================================
# =           LSTM DEFINITIONS                       =
# ====================================================

class BaseLSTM(nn.Module):
    """ Super Simple LSTM with a fully connected layer output (dim=1 output)"""
    def __init__(self, input_size, hidden_size, num_layers, bi=False):
        super().__init__()

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                            bidirectional=bi)
        self.fc = nn.Linear(hidden_size * (int(bi) + 1), 1)

    def forward(self, x):
        return F.sigmoid(self.fc(self.lstm(x)[0]))



class OnlineLSTM(nn.Module):
    """ Online version of LSTM, where inputs are fed in with sequence length of 1 """

    def __init__(self, base_lstm):
        super().__init__()

        self.base_lstm = base_lstm
        self.cn = None
        self.hn = None


    def reset(self):
        self.cn = None
        self.hn = None

    def forward(self, x):
        if self.hn == None:
            out, (hn, cn) = self.base_lstm.lstm(x)
        else:            
            out, (hn, cn) = self.base_lstm.lstm(x, (self.hn, self.cn))
        self.hn = hn
        self.cn = cn
        return F.sigmoid(self.base_lstm.fc(out))



# ===================================================
# =           TCN DEFINITIONs                       =
# ===================================================


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class Conv1dWN(nn.Conv1d):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(Conv1dWN, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias=bias)
        self.g = nn.Parameter(torch.ones(out_channels))

    def forward(self, x):
        wnorm = torch.sqrt(torch.sum(self.weight ** 2))
        return F.conv1d(
            x, 
            self.weight * self.g[:, None, None] / wnorm,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups
            )


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = Conv1dWN(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = Conv1dWN(n_outputs, n_outputs, kernel_size, stride=stride,
                              padding=padding, dilation=dilation)
        #self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
        #                                   stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class BaseTCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super(BaseTCN, self).__init__()
        self.input_size = input_size
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)

    def forward(self, x):
        x = x.view(-1, self.input_size)
        output = self.tcn(x.unsqueeze(0).transpose(1,2)).transpose(1,2)
        output = self.linear(output)
        return torch.nn.functional.sigmoid(output)




class OnlineTCN(nn.Module):
    
    def __init__(self, base_tcn, hist_len, buffer_size=10):
        super().__init__()
        self.base_tcn = base_tcn
        self.hist_len = hist_len
        self.calls = 0
        self.input_size = base_tcn.input_size
        self.buffer_size = buffer_size
        print("ONLINE TCN", self.buffer_size)
        self.reset()
        
    def reset(self):
        device = next(iter(self.base_tcn.parameters())).device
        self.calls = 0
        self.hist = torch.zeros((self.hist_len, self.base_tcn.input_size)).to(device)
        self.buffer = torch.zeros(self.buffer_size).to(self.hist.device)

        
    def forward(self, x):
        self.calls += 1
        self.hist = self.hist.roll(-1, 0)
        self.hist[-1, :] = x.view(1, self.input_size)
        # Two modes:
        if self.calls % self.buffer_size == 0: # update the buffer and return the first element of it
            out = self.base_tcn(self.hist).view(-1)
            self.buffer = out[-self.buffer_size:]
            return self.buffer[0]
        else:
            return self.buffer[self.calls % self.buffer_size]
    
        
    
