import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from MultiHeadAttention import MultiHeadAttention
from FeedForward import FeedForward
from PositionalEncoder import PositionalEncoder

class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_head, dropout = 0.1):
        super.__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_model, d_model, dropout=dropout)
        self.enc_attn = MultiHeadAttention(n_head, d_model, d_model, d_model, dropout=dropout)
        self.ffn = FeedForward(d_model, dropout=dropout)

    def forward(self, dec_input, enc_output,
            slf_attn_mask=None, dec_enc_attn_mask=None):
        dec_output, dec_slf_attn = self.slf_attn(
            dec_input, dec_input, dec_input, mask=slf_attn_mask)
        dec_output, dec_enc_attn = self.enc_attn(
            dec_output, enc_output, enc_output, mask = dec_enc_attn_mask)
        dec_output = self.ffn(dec_output)
        return dec_output, dec_slf_attn, dec_enc_attn

class Decoder(nn.Module):
    def __init___(
            self, n_trg_vocab, d_word_vec, n_layers, n_head, 
            d_model, pad_idx, n_position=200, dropout=0.1, scale_emb=False):
        super().__init__()

        self.trg_word_emb = nn.Embedding(n_trg_vocab, d_word_vec, padding_idx=pad_idx)
        self.position_enc = PositionalEncoder(d_word_vec, n_position=n_position)
        self.dropout = nn.Dropout(dropout)
        self.layer_stack = nn.ModuleList([
            DecoderLayer(d_model, n_head, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.scale_emb = scale_emb
        self.d_model = d_model

    def forward(self, trg_seq, trg_mask, enc_output, src_mask, return_attns=False):
        dec_slf_attn_list, dec_enc_attn_list = []

        dec_output = self.trg_word_emb(trg_seq)
        if self.scale_emb:
            dec_output *= self.d_model ** 0.5
        dec_output = self.dropout(self.position_enc(dec_output))
        dec_output = self.layer_norm(dec_output)

        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn, dec_enc_attn = dec_layer(
                dec_output, enc_output, slf_attn_mask=trg_mask, 
                dec_enc_attn_mask = src_mask
            )
            dec_slf_attn_list += [dec_slf_attn] if return_attns else []
            dec_enc_attn_list += [dec_enc_attn] if return_attns else []

        if return_attns:
            return dec_output, dec_slf_attn_list, dec_enc_attn_list
        return dec_output


