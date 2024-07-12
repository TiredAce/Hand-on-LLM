import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from NormLayer import NormLayer
from MultiHeadAttention import MultiHeadAttention
from FeedForward import FeedForward
from PositionalEncoder import PositionalEncoder

class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout = 0.1):
        super().__init__()

        self.atten = MultiHeadAttention(heads, d_model, d_model, d_model)
        self.ffn = FeedForward(d_model)

    def forward(self, enc_input, mask = None):
        enc_output, enc_slf_attn = self.atten(
            enc_input, enc_input, enc_input, mask
        )
        enc_output = self.ffn(enc_output)
        return enc_output, enc_slf_attn
    

class Encoder(nn.Moudle):
    def __init__(self, n_src_vocab, d_word_vec, n_layers, n_head, 
            d_model, pad_idx, dropout = 0.1, n_position = 200,
            scale_emb = False):
        super().__init__()

        self.src_word_emb = nn.Embedding(n_src_vocab, d_word_vec, padding_idx = pad_idx)
        self.position_enc = PositionalEncoder(d_word_vec, n_position=n_position)
        self.dropout = nn.Dropout(dropout)
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, n_head, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.layer_norm = nn.LayerNorm(d_model, eps = 1e-6)
        self.scale_emb = scale_emb
        self.d_model = d_model

    def forward(self, src_seq, src_mask, return_attns=False):
        enc_slf_attn_list = []

        enc_output = self.src_word_emb(src_seq)
        if self.scale_emb:
            enc_output *= self.d_model ** 0.5
        enc_output = self.dropout(self.position_enc(enc_output))
        enc_output = self.layer_norm(enc_output)

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output, src_mask = src_mask)
            enc_slf_attn_list += [enc_slf_attn] if return_attns else []

        if return_attns:
            return enc_output, enc_slf_attn_list
        
        return enc_output
    