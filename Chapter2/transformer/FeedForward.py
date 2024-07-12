import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class FeedForward(nn.Module):
    def __init__(self, d_model, drop_out = 0.1, eps = 1e-6):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_model * 4)
        self.dropout = nn.Dropout(drop_out)
        self.linear2 = nn.Linear(d_model * 4, d_model)
        self.layer_norm = nn.LayerNorm(d_model, eps = eps)

    def forward(self, x):
        residual = x
        x = self.linear2(F.relu(self.linear1(x)))
        x = self.dropout(x)
        x += residual
        x = self.layer_norm(x)
        return x
    
def main():
    ffn = FeedForward(2)
    input = torch.tensor([[1, 1], [2, 2]], dtype = torch.float32)
    print(ffn(input))

if __name__ == "__main__":
    main()