__author__      = 'gojay'
__date__        = '2026-01-28'
__description__ = 'implementation of sub_layers for encoder-decoder'

import torch
import torch.nn as nn
import torch.nn.functional as F

from sub_layers import MultiHeadAttention, FeedForwardNetwork


class EncoderModule(nn.Module):
    def __init__(self, n_head, d_model, d_k, d_v, d_ff, dropout=0.1):
        super(EncoderModule, self).__init__()

        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

        self.self_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout)
        self.ffn = FeedForwardNetwork(d_model, d_ff, dropout)

    def forward(self, enc_input, mask=None):
        '''
        Args:
            enc_input: input tensor with shape of (B, L, D)
            mask: mask tensor with shape of (B, L, L) or None
            return_attn: whether to return attention tensor
        Returns:
            enc_output: output tensor with shape of (B, L, D)
            attn: attention tensor with shape of (B, N, L, L)
        '''
        x1, attn = self.self_attn(enc_input, enc_input, enc_input, mask)
        x2 = self.layer_norm(enc_input + x1)
        x3 = self.ffn(x2)
        enc_output = self.layer_norm(x2 + x3)
        return enc_output, attn

class DecoderModule(nn.Module):
    def __init__(self):
        super(DecoderModule, self).__init__()

    def forward(self, x):
        return x


if __name__ == '__main__':
    # 0. test encoder module
    #-----------------------------------
    x = torch.randn(1, 256, 768)
    d_model = 768
    n_head = 8
    d_k, d_v = d_model // n_head, d_model // n_head
    d_ff = 2048
    encoder_module = EncoderModule(n_head=n_head, d_model=d_model, d_k=d_k, d_v=d_v, d_ff=d_ff)
    output = encoder_module(x)
    print(output.shape)
