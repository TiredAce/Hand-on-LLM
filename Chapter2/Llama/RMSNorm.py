import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps = 1e-6):
        super.__init__()
        self.hidden_size = hidden_size
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_state):
        input_type = hidden_state.dtype
        variance = hidden_state.to(torch.tensor32).pow(2).mean(-1, keepdim=True)
        hidden_state = hidden_state * torch.rsqrt(variance + self.eps)
        return (self.weight * hidden_state).to(input_type) 

