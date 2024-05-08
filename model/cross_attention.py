import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F


def clones(module, n):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.m = nn.Parameter(torch.ones(features))
        self.n = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.m * (x - mean) / (std + self.eps) + self.n


class AddNorm(nn.Module):
    def __init__(self, size, dropout):
        super(AddNorm, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, a, b):
        return self.norm(a + self.dropout(b))


class EncoderLayer(nn.Module):
    def __init__(self, size, self_attention, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attention = self_attention
        self.feed_forward = feed_forward
        self.sublayer = clones(AddNorm(size, dropout), 2)
        self.size = size

    def forward(self, q, k, v):
        x = self.sublayer[0](v, self.self_attention(q, k, v))
        x = self.sublayer[1](x, self.feed_forward(x))
        return x


class Encoder(nn.Module):

    def __init__(self, layer, n):
        super(Encoder, self).__init__()
        self.layers = clones(layer, n)
        self.layer1 = clones(layer, n)
        self.layer2 = clones(layer, n)

    def forward(self, x_im, x_sk):
        for layer1, layer2 in zip(self.layer1, self.layer2):
            # 在此交换Q
            # layer1 处理草图
            x_sk1 = layer1(x_sk, x_im, x_sk)
            # layer2 处理遥感图像
            x_im = layer2(x_im, x_sk, x_im)
            x_sk = x_sk1
        return x_im, x_sk


def attention(query, key, value, dropout=None):
    """
    dk = dv = dmodel/h = 64,h=8
    """
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)

    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value):
        nbatches = query.size(0)
        # 在batch中进行所有的线性投影（d_model => h x d_k）
        query, key, value = \
            [lin(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for lin, x in zip(self.linears, (query, key, value))]
        # 将注意力应用于所有batch的投影向量中
        x, self.attn = attention(query, key, value, dropout=self.dropout)
        # 最后连接一个线性层
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)

        return self.linears[-1](x)


class PositionWiseFeedForward(nn.Module):
    """
    d_ff = 1024 为论文中数值
    """

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionWiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class CrossAttention(nn.Module):
    def __init__(self, args, h=12, n=1, d_model=768, d_ff=1024, dropout=0.1):
        super(CrossAttention, self).__init__()
        self.args = args
        self.batch = args.batch
        multi_head_attention = MultiHeadedAttention(h, d_model)
        ffn = PositionWiseFeedForward(d_model, d_ff, dropout)
        encoderLayer = EncoderLayer(d_model, multi_head_attention, ffn, dropout)
        self.encoder = Encoder(encoderLayer, n)

    def forward(self, x):
        length = x.size(0)
        x_sk = x[:length // 2]
        x_im = x[length // 2:]
        x_im, x_sk = self.encoder(x_im, x_sk)
        return torch.cat((x_sk, x_im), dim=0)

