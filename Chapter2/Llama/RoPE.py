import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class LllamaRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embedding = 2048,
                 base = 10000, device = None):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(device) / dim))
        self.register_buffer("inv_freq", inv_freq)

        self.max_seq_len_cached = max_position_embedding
        t = torch.arange(self.max_seq_len_cached, device=self.inv_freq.device,
                         dtype=self.inv_freq.dtype)
        