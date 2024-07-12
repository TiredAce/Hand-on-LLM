import torch 
import torch.nn as nn
import math
import numpy as np

class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len = 80):
        super().__init__()
        self.d_model = d_model
        
        # Create a matrix of shape (max_seq_len, d_model) with positional encodings
        position = np.arange(max_seq_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        
        pe = np.zeros((max_seq_len, d_model))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        
        # Convert to PyTorch tensor
        pe = torch.tensor(pe, dtype=torch.float32).unsqueeze(0)
        # Register buffer to avoid updating during training
        self.register_buffer('pe', pe)
	
    def forward(self, x):   # x.shape == (batch_size, seq_size, d_hid)
        return x + self.pe[:, :x.size(1)].clone().detach()
    

def main():
    pe = PositionalEncoder(4)

    token = torch.tensor([[[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3] ]], dtype=torch.float32)
    
    print(pe(token))


if __name__ == "__main__":
    main()