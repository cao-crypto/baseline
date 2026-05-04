"""Self Attention Module


"""
import torch
import torch.nn as nn
import torch.nn.functional as F

import copy
from typing import Optional, Any

import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module
from torch.nn import MultiheadAttention
from torch.nn import ModuleList
from torch.nn.init import xavier_uniform_
from torch.nn import Dropout
from torch.nn import Linear
from torch.nn import LayerNorm
# from nn.module import Module
# from nn.activation import MultiheadAttention
# from nn.container import ModuleList
# from torch.init import xavier_uniform_
# from nn.dropout import Dropout
# from nn.linear import Linear
# from nn.normalization import LayerNorm

def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))


class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model, dim_feedforward=2048, dropout=0.1, activation="relu"):
        # super(nn.TransformerEncoderLayer, self).__init__()
        super(FeedForwardNetwork, self).__init__()
        # self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(nn.TransformerEncoderLayer, self).__setstate__(state)

    def forward(self, src):
        # src2 = self.self_attn(src, src, src, attn_mask=src_mask,
        #                       key_padding_mask=src_key_padding_mask)[0]
        src2 = src
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class TransformerDecoderLayer(Module):
    r"""TransformerDecoderLayer is made up of self-attn, multi-head-attn and feedforward network.
    This standard decoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).

    Examples::
        >>> decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
        >>> memory = torch.rand(10, 32, 512)
        >>> tgt = torch.rand(20, 32, 512)
        >>> out = decoder_layer(tgt, memory)
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerDecoderLayer, self).__setstate__(state)

    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None, memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None, memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the inputs (and mask) through the decoder layer.

        Args:
            tgt: the sequence to the decoder layer (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt


class SelfAttention(nn.Module):
    def __init__(self, in_channel, out_channel=None, attn_dropout=0.1):
        """
        :param in_channel: previous layer's output feature dimension
        :param out_channel: size of output vector, defaults to in_channel
        """
        super(SelfAttention, self).__init__()
        self.in_channel = in_channel

        if out_channel is not None:
            self.out_channel = out_channel
        else:
            self.out_channel = in_channel

        self.temperature = self.out_channel ** 0.5

        self.q_map = nn.Conv1d(in_channel, self.out_channel, 1, bias=False)
        self.k_map = nn.Conv1d(in_channel, self.out_channel, 1, bias=False)
        self.v_map = nn.Conv1d(in_channel, self.out_channel, 1, bias=False)

        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, x):
        """
        :param x: the feature maps from previous layer,
                      shape: (batch_size, in_channel, num_points)
        :return: y: attentioned features maps,
                        shape： (batch_size, out_channel, num_points)
        """
        q = self.q_map(x)  # (batch_size, out_channel, num_points)
        k = self.k_map(x)  # (batch_size, out_channel, num_points)
        v = self.v_map(x)  # (batch_size, out_channel, num_points)

        attn = torch.matmul(q.transpose(1,2) / self.temperature, k)

        attn = self.dropout(F.softmax(attn, dim=-1))
        y = torch.matmul(attn, v.transpose(1,2)) # (batch_size, num_points, out_channel)

        return y.transpose(1, 2)

class CrossAttention(nn.Module):
    def __init__(self, in_channel, out_channel=None, attn_dropout=0.1):
        """
        :param in_channel: previous layer's output feature dimension
        :param out_channel: size of output vector, defaults to in_channel
        """
        super(CrossAttention, self).__init__()
        self.in_channel = in_channel

        if out_channel is not None:
            self.out_channel = out_channel
        else:
            self.out_channel = in_channel

        self.temperature = self.out_channel ** 0.5

        self.q_map = nn.Conv1d(in_channel, self.out_channel, 1, bias=False)
        self.k_map = nn.Conv1d(in_channel, self.out_channel, 1, bias=False)
        self.v_map = nn.Conv1d(in_channel, self.out_channel, 1, bias=False)

        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, x, xx=None):
        """
        :param x: the feature maps from previous layer,
                      shape: (batch_size, in_channel, num_points)
        :return: y: attentioned features maps,
                        shape： (batch_size, out_channel, num_points)
        """
        #  add
        if xx is None:
            xx=x
        q = self.q_map(x)  # (batch_size, out_channel, num_points)
        k = self.k_map(xx)  # (batch_size, out_channel, num_points)
        v = self.v_map(xx)  # (batch_size, out_channel, num_points)

        attn = torch.matmul(q.transpose(1,2) / self.temperature, k)

        attn = self.dropout(F.softmax(attn, dim=-1))
        y = torch.matmul(attn, v.transpose(1,2)) # (batch_size, num_points, out_channel)

        return y.transpose(1, 2)


class MultiHeadAttention(nn.Module):
    def __init__(self, in_channel, out_channel, n_classes=2, n_heads=1, att_dropout=0.1, use_proj=True):
        """
        :param in_channel: previous layer's output feature dimension
        :param out_channel: size of output vector, defaults to in_channel
        """
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.temperature = (out_channel // self.n_heads) ** 0.5
        self.n_classes = n_classes
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.use_proj = use_proj

        self.q_map = nn.Linear(self.in_channel, self.out_channel)
        self.k_map = nn.Linear(self.in_channel, self.out_channel)
        self.v_map = nn.Linear(self.in_channel, self.out_channel)
        self.dropout = nn.Dropout(att_dropout)

        if self.use_proj:
            self.proj = nn.Sequential(nn.Linear(self.out_channel // 2, self.out_channel // 2),
                                      nn.ReLU(inplace=True),
                                      nn.Linear(self.out_channel // 2, self.out_channel))

    def forward(self, x):
        """
        :param x: the feature maps from previous layer,
                      shape: (batch_size, in_channel, num_points)
        :param mask: shape(B, N, 1)
        :return: y: attentioned features maps,
                        shape： (batch_size, out_channel, num_points)
        """

        q, k, v = x

        B, N = q.shape[0], q.shape[1]

        q_res = q
        q = self.q_map(q)
        if q_res.size(-1) != q.size(-1):
            q_res = q
        k = self.k_map(k)
        v = self.v_map(v)

        q = q.reshape(B, N, self.n_heads, self.out_channel // self.n_heads).permute(0, 2, 1, 3)
        k = k.reshape(k.shape[0], k.shape[1], self.n_heads, self.out_channel // self.n_heads).permute(0, 2, 1, 3)
        v = v.reshape(v.shape[0], v.shape[1], self.n_heads, self.out_channel // self.n_heads).permute(0, 2, 1, 3)

        # [n_head, B, N, B*N]
        attn = torch.matmul(q, k.transpose(2, 3)) / self.temperature  # [1, 4, 100, 4096]
        attn = self.dropout(F.softmax(attn, dim=-1))
        y = torch.matmul(attn, v).transpose(1, 2).reshape(B, N, -1)

        if self.use_proj:
            y = self.proj(y)

        y = y + q_res
        return y

# original QGPA
class QGPA(nn.Module):
    def __init__(self, attn_dropout=0.1):

        super(QGPA, self).__init__()
        self.in_channel = self.out_channel = 320

        self.temperature = self.out_channel ** 0.5
        self.layer_norm = nn.LayerNorm(self.in_channel)
        proj_dim = 512
        self.q_map = nn.Conv1d(2048, proj_dim, 1, bias=False)
        self.k_map = nn.Conv1d(2048, proj_dim, 1, bias=False)

        self.v_map = nn.Linear(self.in_channel, self.out_channel)
        self.fc = nn.Conv1d(self.in_channel, self.out_channel, 1, bias=False)
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, query, support, prototype):
        '''
        :param query: [2, 320, 2048]
        :param support: [2, 320, 2048]
        :param prototype: [2, 3, 320]
        :return: [2, 3, 320]
        '''

        batch, dim = query.shape[0], query.shape[1]
        way = support.shape[0] + 1
        residual = prototype
        q = self.q_map(query.transpose(1, 2))  # [2, 512, 320]  2way,2048points-512,320fea_dim
        if len(support.shape) == 4:
            support = support.squeeze()
        support = torch.cat([support.mean(0).unsqueeze(0), support], dim=0)  # [3, 512, 320]
        k = self.k_map(support.transpose(1, 2))  # [3, 512, 320]
        v = self.v_map(prototype)  # [2, 3, 320]
        q = q.view(q.shape[1], q.shape[2] * q.shape[0])  # [512, 640]
        k = k.view(k.shape[1], k.shape[2] * k.shape[0])  # [512, 960]

        attn = torch.matmul(q.transpose(0, 1) / self.temperature, k)  # [640, 960]
        attn = attn.reshape(batch, way, dim, dim)  # [2, 3, 320, 320]
        attn = F.softmax(attn, dim=-1)
        v = v.unsqueeze(2)
        output = torch.matmul(attn, v.transpose(-2, -1)).squeeze(-1).transpose(1, 2)
        output = self.dropout(self.fc(output)).transpose(1, 2)
        output = self.layer_norm(output + residual)

        return output