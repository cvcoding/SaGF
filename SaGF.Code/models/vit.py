# https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit_pytorch.py

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn
from functools import partial
from itertools import repeat
from collections import OrderedDict
from einops import rearrange
from einops.layers.torch import Rearrange
import math
from pygcn.models import GCN, GIN
from pygcn.models_gru import GCN_gru
import numpy as np
# import cupy as cp
from models import *
from torch.autograd import Variable
import torch.nn.init as init
from torch import Tensor
from typing import Tuple
from models.triattention import TripletAttention


dtype = torch.cuda.FloatTensor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# from torch._six import container_abcs

# From PyTorch internals
def _ntuple(n):
    def parse(x):
        # if isinstance(x, container_abcs.Iterable):
        #     return x
        return tuple(repeat(x, n))

    return parse


to_2tuple = _ntuple(2)

MIN_NUM_PATCHES = 16


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        temp = self.norm(x)
        return self.fn(temp, *args, **kwargs)


# class PreForward(nn.Module):
#     def __init__(self, dim, hidden_dim, kernel_size, num_channels, dropout=0.):
#         super().__init__()
#         # self.net = nn.Sequential(
#         #     nn.Linear(dim, hidden_dim),
#         #     nn.GELU(),
#         #     nn.Dropout(dropout),
#         #     nn.Linear(hidden_dim, dim),
#         #     nn.Dropout(dropout)
#         # )
#         self.tcn = TemporalConvNet(dim, num_channels, hidden_dim, kernel_size, dropout)
#         # self.net = nn.Sequential(
#         #     nn.Linear(dim, hidden_dim),
#         #     nn.GELU(),
#         #     nn.Dropout(dropout),
#         # )
#
#     def forward(self, x):
#         r = self.tcn(x.permute(0, 2, 1)).permute(0, 2, 1)
#         # r = self.net(r)
#         return r


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, image_size, patch_size, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            # nn.Dropout(dropout)
        )
        # self.net = nn.Identity()
    def forward(self, x):
        return self.net(x)


# def inverse_gumbel_cdf(y, mu, beta):
#     return mu - beta * torch.log(-torch.log(y))
class TopkRouting(nn.Module):
    """
    differentiable topk routing with scaling
    Args:
        qk_dim: int, feature dimension of query and key
        topk: int, the 'topk'
        qk_scale: int or None, temperature (multiply) of softmax activation
        with_param: bool, wether inorporate learnable params in routing unit
        diff_routing: bool, wether make routing differentiable
        soft_routing: bool, wether make output value multiplied by routing weights
    """

    def __init__(self, topk=4, diff_routing=False):
        super().__init__()
        self.topk = topk
        self.diff_routing = diff_routing
        # routing activation
        self.routing_act = nn.Softmax(dim=-1)

    def forward(self, gen_adj: Tensor) -> Tuple[Tensor]:
        """
        Args:
            q, k: (n, p^2, c) tensor
        Return:
            r_weight, topk_index: (n, p^2, topk) tensor
        """
        topk_attn_logit, topk_index = torch.topk(gen_adj, k=self.topk, dim=-1)
        r_weight = self.routing_act(topk_attn_logit)

        # attn_logit = torch.mean(torch.abs(noise), dim=-1)  # (n, p^2, p^2)
        # topk_attn_logit, topk_index = torch.topk(attn_logit, k=self.topk, dim=-1)  # (n, p^2, k), (n, p^2, k)
        # r_weight = self.routing_act(topk_attn_logit)  # (n, p^2, k)

        return r_weight, topk_index


class KVGather(nn.Module):
    def __init__(self, mul_weight='none'):
        super().__init__()
        assert mul_weight in ['none', 'soft', 'hard']
        self.mul_weight = mul_weight

    def forward(self, r_idx: Tensor, r_weight: Tensor, kv: Tensor):
        """
        r_idx: (n, p^2, topk) tensor
        r_weight: (n, p^2, topk) tensor
        kv: (n, p^2, w^2, c_kq+c_v)
        Return:
            (n, p^2, topk, w^2, c_kq+c_v) tensor
        """
        # select kv according to routing index
        n, p2, w2, c_kv = kv.size()
        topk = r_idx.size(-1)
        # print(r_idx.size(), r_weight.size())
        # FIXME: gather consumes much memory (topk times redundancy), write cuda kernel?
        mm = kv.view(n, 1, p2, w2, c_kv).expand(-1, p2, -1, -1, -1)
        topk_kv = torch.gather(mm,
                               # (n, p^2, p^2, w^2, c_kv) without mem cpy
                               dim=2,
                               index=r_idx.view(n, p2, topk, 1, 1).expand(-1, -1, -1, w2, c_kv)
                               # (n, p^2, k, w^2, c_kv)
                               )

        if self.mul_weight == 'soft':
            topk_kv = r_weight.view(n, p2, topk, 1, 1) * topk_kv  # (n, p^2, k, w^2, c_kv)
        elif self.mul_weight == 'hard':
            raise NotImplementedError('differentiable hard routing TBA')
        # else: #'none'
        #     topk_kv = topk_kv # do nothing

        return topk_kv


class KGather(nn.Module):
    def __init__(self, mul_weight='none'):
        super().__init__()
        assert mul_weight in ['none', 'soft', 'hard']
        self.mul_weight = mul_weight

    def forward(self, r_idx:Tensor, r_weight:Tensor, k:Tensor):
        """
        r_idx: (n, p^2, topk) tensor
        r_weight: (n, p^2, topk) tensor
        kv: (n, p^2, w^2, c_kq+c_v)

        Return:
            (n, p^2, topk, w^2, c_kq+c_v) tensor
        """
        # select kv according to routing index
        n, p2, w2, c_k = k.size()
        topk = r_idx.size(-1)
        # print(r_idx.size(), r_weight.size())
        # FIXME: gather consumes much memory (topk times redundancy), write cuda kernel?
        topk_k = torch.gather(k.view(n, 1, p2, w2, c_k).expand(-1, p2, -1, -1, -1), # (n, p^2, p^2, w^2, c_kv) without mem cpy
                                dim=2,
                                index=r_idx.view(n, p2, topk, 1, 1).expand(-1, -1, -1, w2, c_k) # (n, p^2, k, w^2, c_kv)
                               )

        if self.mul_weight == 'soft':
            topk_k = r_weight.view(n, p2, topk, 1, 1) * topk_k # (n, p^2, k, w^2, c_kv)
        elif self.mul_weight == 'hard':
            raise NotImplementedError('differentiable hard routing TBA')

        return topk_k


class Attention_local(nn.Module):
    def __init__(self,
                 depth,
                 i,
                 dim,
                 image_size,
                 patch_size,
                 heads=8,
                 dropout=0,
                 qkv_bias=False,
                 attn_drop=0.,
                 proj_drop=0.,
                 downsample=0.,
                 kernel_size=5,
                 stride_kv=1,
                 stride_q=1,
                 padding_kv=2,
                 padding_q=2,
                 method='dw_bn',
                 with_cls_token=False
                 ):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5

        # self.to_qkv = nn.Linear(dim, dim * 3, bias = False)
        # self.to_out = nn.Sequential(
        #     nn.Linear(dim, dim),
        #     nn.Dropout(dropout)
        # )

        self.stride_kv = stride_kv
        self.stride_q = stride_q
        self.dim = dim
        self.num_heads = heads
        self.scale = dim ** -0.5
        self.with_cls_token = with_cls_token

        dim_in = dim
        dim_out = dim
        self.conv_proj_q = self._build_projection(
            dim_in, dim_out, kernel_size, padding_q,
            stride_q, 'linear' if method == 'avg' else method
        )
        self.conv_proj_k = self._build_projection(
            dim_in, dim_out, kernel_size, padding_kv,
            stride_kv, method
        )
        self.conv_proj_v = self._build_projection(
            dim_in, dim_out, kernel_size, padding_kv,
            stride_kv, method
        )

        # self.proj_q = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        # self.proj_k = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        # self.proj_v = nn.Linear(dim_in, dim_out, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim_out, dim_out)
        # self.proj1 = nn.Linear(dim_out, 64)
        # self.proj2 = nn.Linear(64, dim_out)
        self.proj_drop = nn.Dropout(proj_drop)
        self.k_gather = KVGather(mul_weight='none')
        self.attn_act = nn.Softmax(dim=-1)

    def _build_projection(self,
                          dim_in,
                          dim_out,
                          kernel_size,
                          padding,
                          stride,
                          method):
        if method == 'dw_bn':
            proj = nn.Sequential(OrderedDict([
                ('conv', nn.Conv2d(
                    dim_in,
                    dim_in,
                    kernel_size=kernel_size,
                    padding=padding,
                    stride=stride,
                    bias=False,
                    groups=dim_in
                )),
                ('bn', nn.BatchNorm2d(dim_in)),
                ('rearrage', Rearrange('b c h w -> b (h w) c')),
            ]))
        elif method == 'avg':
            proj = nn.Sequential(OrderedDict([
                ('avg', nn.AvgPool2d(
                    kernel_size=kernel_size,
                    padding=padding,
                    stride=stride,
                    ceil_mode=True
                )),
                ('rearrage', Rearrange('b c h w -> b (h w) c')),
            ]))
        elif method == 'linear':
            proj = None
        else:
            raise ValueError('Unknown method ({})'.format(method))

        return proj

    def forward_conv(self, x, h, w):
        if self.with_cls_token:
            cls_token, x = torch.split(x, [1, h * w], 1)

        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

        if self.conv_proj_q is not None:
            q = self.conv_proj_q(x)
        else:
            q = rearrange(x, 'b c h w -> b (h w) c')

        if self.conv_proj_k is not None:
            k = self.conv_proj_k(x)
        else:
            k = rearrange(x, 'b c h w -> b (h w) c')

        if self.conv_proj_v is not None:
            v = self.conv_proj_v(x)
        else:
            v = rearrange(x, 'b c h w -> b (h w) c')

        if self.with_cls_token:
            q = torch.cat((cls_token, q), dim=1)
            k = torch.cat((cls_token, k), dim=1)
            v = torch.cat((cls_token, v), dim=1)

        return q, k, v

    def forward(self, x, noise, gen_adj, sparsity):
        b, L, d = x.size()
        h = int(math.sqrt(L))
        w = h
        # tm = torch.sum(torch.abs(noise), dim=-1)
        # tm_max, _ = torch.max(tm, dim=-1, keepdim=True)
        # tm_min, _ = torch.min(tm, dim=-1, keepdim=True)
        # tm = (tm - tm_min)/(tm_max - tm_min)
        # noise = torch.tanh(tm).unsqueeze(2).expand(-1, -1, d)
        # x = x * noise
        q, k, v = self.forward_conv(x, h, w)
        q = rearrange(q, 'b t (h d) -> b h t d', h=self.num_heads)
        k = rearrange(k, 'b t (h d) -> b h t d', h=self.num_heads)
        v = rearrange(v, 'b t (h d) -> b h t d', h=self.num_heads)

        # adj = rearrange(adj_copy, '(b h) t d -> b h t d', h=self.num_heads)

        kv = torch.cat((k, v), dim=-1)
        # kv_pix = rearrange(kv, 'n h t c -> n t (h c)')

        self.router = TopkRouting(topk=sparsity, diff_routing=False)

        r_weight, r_idx = self.router(gen_adj)

        # r_idx_expanded = r_idx.unsqueeze(2).expand(-1, -1, d*2)
        # r_weight_expanded = r_weight.unsqueeze(2).expand(-1, -1, d * 2)
        # kv_pix_sel = torch.gather(kv_pix, 1, r_idx_expanded) * r_weight_expanded

        q = rearrange(q, 'n e (h w) c -> (n e) h w c', h=h, w=w)
        q_pix = rearrange(q, 'n (i h) (j w) c -> n (h w) (i j) c', i=2, j=2)
        kv = rearrange(kv, 'n e (h w) c -> (n e) h w c', h=h, w=w)
        kv_pix = rearrange(kv, 'n (i h) (j w) c -> n (h w) (i j) c', i=2, j=2)

        # kv_pix = rearrange(kv, 'n h t c -> (n h) t c').unsqueeze(2)
        r_weight, r_idx = rearrange(r_weight, 'n h t c -> (n h) t c'), rearrange(r_idx, 'n h t c -> (n h) t c')
        kv_pix_sel = self.k_gather(r_idx=r_idx, r_weight=r_weight, kv=kv_pix)  # (n, p^2, topk, h_kv*w_kv, c_qk+c_v)

        k_pix_sel, v_pix_sel = kv_pix_sel.split([self.dim//self.num_heads, self.dim//self.num_heads], dim=-1)

        k_pix_sel = rearrange(k_pix_sel, '(n m) p2 k w2 c -> (n p2) m c (k w2)',
                              m=self.num_heads)  # flatten to BMLC, (n*p^2, m, topk*h_kv*w_kv, c_kq//m) transpose here?
        v_pix_sel = rearrange(v_pix_sel, '(n m) p2 k w2 c -> (n p2) m (k w2) c',
                              m=self.num_heads)  # flatten to BMLC, (n*p^2, m, topk*h_kv*w_kv, c_v//m)
        q_pix = rearrange(q_pix, '(n m) p2 w2 c -> (n p2) m w2 c',
                          m=self.num_heads)  # to BMLC tensor (n*p^2, m, w^2, c_qk//m)

        # param-free multihead attention
        attn_weight = (q_pix * self.scale) @ k_pix_sel


        # k_pix_sel = rearrange(k_pix_sel.squeeze(), '(n h) t k c -> (n t) h  c k', h=self.num_heads)
        # v_pix_sel = rearrange(v_pix_sel.squeeze(), '(n h) t k c -> (n t) h  k c', h=self.num_heads)
        # q2 = rearrange(q, 'n h t c -> (n t) h c').unsqueeze(2)
        # attn_weight = (q2 * self.scale) @ k_pix_sel

        # attn_weight = (q * self.scale) @ k_pix_sel.transpose(-2, -1)

        # k_pix_sel = rearrange(k_pix_sel, '(n h) t k w2 c -> (n t) h  c (k w2)', h=self.num_heads)
        # v_pix_sel = rearrange(v_pix_sel, '(n h) t k w2 c -> (n t) h  (k w2) c', h=self.num_heads)
        # q2 = rearrange(q, 'n h t c -> (n t) h c').unsqueeze(2)

        attn_weight = self.attn_act(attn_weight)

        v = attn_weight @ v_pix_sel

        v = rearrange(v, '(n h w) m (i j) c -> n (j h) (i w) (m c)', j=2, i=2, h=h//2, w=w//2)

        v = (rearrange(v, 'n w h c -> n (w h) c'))

        out = v  # self.proj_v(v)

        return out


class ConvEmbed(nn.Module):
    """ Image to Conv Embedding

    """

    def __init__(self,
                 image_size,
                 patch_size,
                 kernel_size,
                 batch_size,
                 in_chans,
                 embed_dim,
                 stride,
                 padding,
                 norm_layer=None):
        super().__init__()

        self.proj = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(
                in_chans, int(embed_dim),
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                padding_mode='replicate',  #reflect  zeros  circular  replicate
                # groups=in_chans
            )),
            ('bn', nn.BatchNorm2d(int(embed_dim))),
            ('relu', nn.GELU()),
            ('pooling', nn.MaxPool2d(kernel_size=3, stride=stride, padding=1,)),
            # ('pooling', nn.AdaptiveMaxPool2d((3, 3))),
            # ('bn', nn.BatchNorm2d(int(embed_dim))),
            # ('relu', nn.GELU()),
        ]))

    def forward(self, x):
        sp_features = self.proj(x).to(device)  # proj_conv  proj

        return sp_features


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout, image_size, patch_size, kernel_size, downsample, batch_size, in_chans,
                 patch_stride, patch_padding, norm_layer=nn.LayerNorm):
        super().__init__()

        self.patch_embed = ConvEmbed(
            image_size=image_size,
            patch_size=patch_size,
            kernel_size=kernel_size,
            batch_size=batch_size,
            in_chans=in_chans,
            stride=patch_stride,
            padding=1,
            embed_dim=dim//8,
            norm_layer=norm_layer
        )
        self.patch_dim = (int(patch_size//4) ** 2) * int(dim)//8 #((patch_size // 4) ** 2) * int(dim) // 4
        self.dim = dim
        #
        # channels = 3
        # self.patch_dim = channels * patch_size ** 2

        self.patch_to_embedding = nn.Linear(self.patch_dim, dim).to(device)

        self.layers = nn.ModuleList([])
        self.depth = depth
        self.depth4pool = depth + 1   #1.5
        self.patchnum = image_size//patch_size

        self.depth4pool2 = int(depth + 1)  # 1.5
        for i in range(self.depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim * 2,
                                 Attention_local(depth, i, dim * 2, image_size=image_size, patch_size=patch_size,
                                                 heads=heads, dropout=dropout, downsample=downsample,
                                                 kernel_size=kernel_size))),
                # PreNorm(dim*2, Residual(Attention_local(depth, i, dim*2, image_size=image_size, patch_size=patch_size, heads=heads, dropout=dropout, downsample=downsample, kernel_size=kernel_size))),
                # FeedForward(dim*2, mlp_dim*2, image_size, patch_size, dropout=dropout),
                Residual(PreNorm(dim*2, FeedForward(dim*2, mlp_dim*2, image_size, patch_size, dropout=dropout))),
            ]))

        # for i in range(depth-self.depth4pool):
        #     self.layers.append(nn.ModuleList([
        #         # Residual(PreNorm(dim, Attention(depth, i, dim, image_size=image_size, patch_size=patch_size, heads=heads, dropout=dropout, downsample=downsample, kernel_size=kernel_size))),
        #         PreNorm(dim, Residual(Attention_global(depth, i, dim, image_size=image_size, patch_size=patch_size, heads=heads, dropout=dropout, downsample=downsample, kernel_size=kernel_size))),
        #         FeedForward(dim, mlp_dim, image_size, patch_size, dropout=dropout),
        #     ]))

        self.dropout = nn.Dropout(dropout)

        # self.norm = nn.ModuleList([])
        # for _ in range(depth):
        #     self.norm.append(nn.LayerNorm(dim))

        self.patch_size = patch_size
        self.batch_size = batch_size
        self.head_num = heads
        # UT = torch.randn((int(image_size/patch_size*downsample)**2, dim), requires_grad=True).to(device)
        # self.UT = torch.nn.Parameter(UT)
        # self.register_parameter("Ablah2", self.UT)

        # UT2 = torch.randn((int(image_size / patch_size * downsample * downsample) ** 2, dim), requires_grad=True).to(device)
        # self.UT2 = torch.nn.Parameter(UT2)
        # self.register_parameter("Ablah3", self.UT2)

        # self.Upool = nn.Sequential(
        #     nn.Linear(dim, int(image_size/patch_size*downsample)**2, bias=True),
        #     # nn.Dropout(dropout)
        # )
        # self.Upool2 = nn.Sequential(
        #     nn.Linear(dim, int(image_size / patch_size * downsample * 0.5) ** 2, bias=True),
        #     # nn.Dropout(dropout)
        # )

        # Upool_out = torch.randn((1, dim), requires_grad=True).to(device)
        # self.Upool_out = torch.nn.Parameter(Upool_out)
        # self.register_parameter("Ablah4", self.Upool_out)
        self.Upool_out = nn.Sequential(
            nn.Linear(dim*2, 1, bias=False),)
        self.triattention = TripletAttention()
        # self.Upool_inter = nn.Sequential(
        #     nn.Linear(dim, 1, bias=False), )

    def forward(self, img, x_pre, x_lowrank, noise, gen_adj, sparsity):  #lowrank 选一个作为全局节点，传给vit 的attention
        # p = self.patch_size
        # x = rearrange(img, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=p, p2=p)
        # x = self.patch_to_embedding(x)
        # x = torch.cat([x, gen_x], dim=-1)
        # b, n, _ = x.shape

        p = self.patch_size
        b, n, imgh, imgw = img.shape

        x = rearrange(img, 'b c (h p1) (w p2) -> (b h w) (c) (p1) (p2)', p1=p, p2=p)
        conv_img = self.patch_embed(x)
        conv_img = rearrange(conv_img, '(b s) c p1 p2 -> b s (c p1 p2)', b=b)
        # x = rearrange(img, 'b c (h p1) (w p2) -> b (c p1 p2) h w', p1=p, p2=p)
        # conv_img = self.triattention(x)
        # conv_img = rearrange(conv_img, 'b c h w -> b (h w) c')

        x = self.patch_to_embedding(conv_img)

        x = rearrange(x.permute(0, 2, 1), 'b c (w h) -> b c w h', w=self.patchnum, h=self.patchnum)

        x = self.triattention(x)
        x = rearrange(x, 'b c w h -> b c (w h)').permute(0, 2, 1)

        x = torch.cat([x, x_pre], dim=-1)
        b, n, _ = x.shape

        index = 0
        for attn, ff in self.layers:
            # x = attn(x, self.rep_adj, 0)
            if index < self.depth4pool:
                x = attn(x, noise, gen_adj, sparsity)
                x = ff(x)
            # else:
            #     if index == self.depth4pool:
            #         # temp = torch.matmul(self.UT, x.permute(0, 2, 1))
            #         temp = self.Upool(x).permute(0, 2, 1)
            #         C = F.gumbel_softmax(temp, dim=-1, tau=1.0)
            #         # C = F.softmax(temp/2, dim=-1)
            #         x = torch.matmul(C, x)
            #         C = C.unsqueeze(dim=1).expand(b, self.head_num, -1, -1).to(device)
            #         temp2 = torch.matmul(C, rep_adj)
            #         rep_adj = torch.matmul(temp2, C.permute(0, 1, 3, 2))
            #
            #     if index == self.depth4pool2:
            #         # temp = torch.matmul(self.UT2, x.permute(0, 2, 1))
            #         temp = self.Upool2(x).permute(0, 2, 1)
            #         C = F.gumbel_softmax(temp, dim=-1, tau=1.0)
            #         # C = F.softmax(temp/2, dim=-1)
            #         x = torch.matmul(C, x)
            #         C = C.unsqueeze(dim=1).expand(b, self.head_num, -1, -1).to(device)
            #         temp2 = torch.matmul(C, rep_adj)
            #         rep_adj = torch.matmul(temp2, C.permute(0, 1, 3, 2))
            #     x = attn(x, rep_adj, rep_adj_dis)
            #     x = ff(x)

            index = index + 1
        # temp = torch.matmul(self.Upool_out, x.permute(0, 2, 1))
        temp = self.Upool_out(x).permute(0, 2, 1)
        # temp = F.gumbel_softmax(temp, dim=-1, tau=1.0)
        # temp = F.softmax(temp/2, -1)
        temp = F.normalize(temp, dim=-1)
        x_out = torch.matmul(temp, x)
        # x_out = torch.cat((x_out, x_inter), dim=-1)
        # x_out = F.normalize(x, dim=-1)
        return x_out


class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, kernel_size, downsample, batch_size, num_classes, dim, depth, heads,
                 mlp_dim, patch_stride, patch_pading, in_chans, dropout=0., emb_dropout=0., expansion_factor=1):
        super().__init__()
        assert image_size % patch_size == 0, 'image dimensions must be divisible by the patch size'

        self.patch_size = patch_size

        # self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        # self.patch_to_embedding = nn.Linear(patch_dim, dim)
        # self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, mlp_dim, dropout, image_size, patch_size, kernel_size, downsample,
                                       batch_size, in_chans, patch_stride=patch_stride, patch_padding=patch_pading)

        self.to_cls_token = nn.Identity()
        self.heads = heads
        # self.projection = MLP(dim, dim)
        self.predictor = Predictor(dim, num_classes)

    def forward(self, img, x_pre, x_lowrank, noise, gen_adj, sparsity):

        # if interation is not None and interation %3 ==0:
        #     alpha = 1 - torch.tanh(torch.tensor(interation/16))  # total epoch /2 +1
        #     alpha = int(alpha)
        #     adj_matrix = alpha * adj_matrix + (1-alpha)*gen_adj

        x = self.transformer(img, x_pre, x_lowrank, noise, gen_adj, sparsity)
        # x = self.to_cls_token(x[:, -1])
        pred = self.to_cls_token(x.squeeze())
        class_result = self.predictor(x.squeeze())
        return pred, class_result


# class MLP(nn.Module):
#     def __init__(self, dim):
#         super().__init__()
#         hidden_size = dim
#         self.net = nn.Sequential(
#             nn.Linear(dim*2, hidden_size),
#             nn.BatchNorm1d(hidden_size),
#             # nn.ReLU(inplace=True),
#             # nn.Linear(hidden_size, projection_size)
#         )
#
#     def forward(self, x):
#         return self.net(x)


class Predictor(nn.Module):
    def __init__(self, dim, num_classes):
        super().__init__()
        hidden_size = dim
        self.mlp_head = nn.Sequential(
            # nn.Linear(dim, hidden_size),
            # nn.BatchNorm1d(hidden_size),
            # nn.ReLU(),
            # nn.Linear(hidden_size, num_classes),
            nn.Linear(dim*2, num_classes)
        )

    # def init(self):
    #     init.xavier_uniform_(self.mlp_head[0].weight)  # 例如，使用 Kaiming Normal 初始化
    #     init.constant_(self.mlp_head[0].bias, 0)  # 例如，将所有偏置设置为 0
    #     init.xavier_uniform_(self.mlp_head[3].weight)  # 例如，使用 Kaiming Normal 初始化
    #     init.constant_(self.mlp_head[3].bias, 0)  # 例如，将所有偏置设置为 0

    def forward(self, x):
        return self.mlp_head(x)
