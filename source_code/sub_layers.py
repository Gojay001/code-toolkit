__author__      = 'gojay'
__date__        = '2026-01-28'
__description__ = 'implementation of sub_layers for encoder-decoder'

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        '''
        Args:
            q, k ,v: query tensor with shape of (B, N, L, D)
            mask: mask tensor with shape of (B, N, L, L) or None
        Returns:
            output: output tensor with shape of (B, N, L, D)
            attn: attention tensor with shape of (B, N, L, L)
        '''
        attn = torch.matmul(q, k.transpose(2, 3))
        d_k = k.size(-1)
        attn = attn / d_k ** 0.5

        if mask is not None:
            attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn

class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super(MultiHeadAttention, self).__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_q = nn.Linear(d_model, d_k * n_head, bias=False)
        self.w_k = nn.Linear(d_model, d_k * n_head, bias=False)
        self.w_v = nn.Linear(d_model, d_v * n_head, bias=False)
        self.w_o = nn.Linear(d_v * n_head, d_model, bias=False)

        self.attetion = ScaledDotProductAttention(dropout)

    def forward(self, q, k, v, mask=None):
        '''
        Args:
            q, k, v: query, key, value tensor with shape of (B L, D)
            mask: mask tensor with shape of (B L, L) or None
        Returns:
            output: output tensor with shape of (B, L, D)
            attn: attention tensor with shape of (B, N, L, L)
        '''
        bs, len_q, len_k, len_v = q.shape[0], q.shape[1], k.shape[1], v.shape[1]

        # view to (B, L, N, D)
        q = self.w_q(q).view(bs, len_q, self.n_head, self.d_k)
        k = self.w_k(k).view(bs, len_k, self.n_head, self.d_k)
        v = self.w_v(v).view(bs, len_v, self.n_head, self.d_v)

        # transpose to (B, N, L, D)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)

        # apply scaled dot product attention
        output, attn = self.attetion(q, k, v, mask)

        output = output.transpose(1, 2).contiguous().view(bs, len_q, -1)
        output = self.w_o(output)

        return output, attn

class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(FeedForwardNetwork, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        '''
        Args:
            x: input tensor with shape of (B, L, D)
        Returns:
            output: output tensor with shape of (B, L, D)
        '''
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_pos=2000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_pos, d_model)

        pos = torch.arange(max_pos, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0.0, d_model, 2.0) * -(math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        '''
        Args:
            x: input tensor with shape of (B, L, D)
        Returns:
            output: output tensor with shape of (B, L, D)
        '''
        x = x + self.pe[:, :x.size(1), :].to(x.device)
        return x


if __name__ == '__main__':
    # 0. test positional encoding
    #-----------------------------------
    x = torch.randn(1, 256, 768)
    pe = PositionalEncoding(d_model=768)
    output = pe(x)
    print(output.shape)
    print(output[0, 0, :10])

    # 1. test scaled dot product attention
    #-----------------------------------
    # q = torch.randn(1, 8, 256, 768)
    # k = torch.randn(1, 8, 256, 768)
    # v = torch.randn(1, 8, 256, 768)
    # attention = ScaledDotProductAttention()
    # output, attn = attention(q, k, v)
    # print(output.shape, attn.shape)

    # 2. test multi-head attention
    #-----------------------------------
    # q = torch.randn(1, 256, 768)
    # k = torch.randn(1, 256, 768)
    # v = torch.randn(1, 256, 768)
    # d_model = 768
    # n_head = 8
    # d_k, d_v = d_model // n_head, d_model // n_head
    # attention = MultiHeadAttention(n_head=n_head, d_model=d_model, d_k=d_k, d_v=d_v)
    # output, attn = attention(q, k, v)
    # print(output.shape, attn.shape)

    # 3. test feed forward network
    #-----------------------------------
    # # x = torch.randn(1, 256, 768)
    # feed_forward = FeedForwardNetwork(d_model=768, d_ff=2048)
    # output = feed_forward(output)
    # print(output.shape)
