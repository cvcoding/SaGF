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
from utils import *
import math
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
EOS = 1e-15


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
        att, adj = self.fn(x, *args, **kwargs)
        out = att + x, adj
        return out


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        temp = self.norm(x)
        out = self.fn(temp, *args, **kwargs)
        return out

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, image_size, patch_size, dropout=0.):
        super().__init__()
        # self.net = nn.Sequential(
        #     nn.Linear(dim, hidden_dim),
        #     nn.GELU(),
        #     nn.Dropout(dropout),
        #     # nn.Linear(hidden_dim, dim),
        #     # nn.Dropout(dropout)
        # )
        self.net = nn.Identity()
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

    def forward(self, adj: Tensor) -> Tuple[Tensor]:
        """
        Args:
            q, k: (n, p^2, c) tensor
        Return:
            r_weight, topk_index: (n, p^2, topk) tensor
        """

        attn_logit = adj  # (n, p^2, p^2)
        topk_attn_logit, topk_index = torch.topk(attn_logit, k=self.topk, dim=-1)  # (n, p^2, k), (n, p^2, k)
        r_weight = self.routing_act(topk_attn_logit)  # (n, p^2, k)

        return r_weight, topk_index


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


class Attention_3GIN(nn.Module):
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

        self.proj_q = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.proj_k = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.proj_v = nn.Linear(dim_in, dim_out, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim_out, dim_out)
        # self.proj1 = nn.Linear(dim_out, 64)
        # self.proj2 = nn.Linear(64, dim_out)
        self.proj_drop = nn.Dropout(proj_drop)
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


    def forward(self, x):

        b, L, d = x.size()
        h = int(math.sqrt(L))
        w = h
        q, k, v = self.forward_conv(x, h, w)
        q = rearrange(q, 'b t (h d) -> b h t d', h=self.num_heads)
        k = rearrange(k, 'b t (h d) -> b h t d', h=self.num_heads)
        v = rearrange(v, 'b t (h d) -> b h t d', h=self.num_heads)

        # x = rearrange(x, 'b t (h d) -> b h t d', h=self.num_heads)

        # q, k, v, _ = self.forward_conv_qkv(x, adj)
        q_pix = q #rearrange(q, 'n h t c -> (n t) h c').unsqueeze(2)
        k_pix = rearrange(k, 'n h t c -> n h c t')
        v_pix = v #rearrange(v, 'n h t c -> (n t) h c').unsqueeze(2)

        attn_weight = (q_pix * self.scale) @ k_pix  # (n*p^2, m, w^2, c) @ (n*p^2, m, c, topk*h_kv*w_kv) -> (n*p^2, m, w^2, topk*h_kv*w_kv)

        attn_weight = self.attn_act(attn_weight)
        v = attn_weight @ v_pix

        v = F.gelu((rearrange(v, 'b h t d -> b t (h d)', h=self.num_heads)))

        out = (self.proj_v(v)) #torch.sigmoid

        return out, attn_weight


# class ConvEmbed(nn.Module):
#     """ Image to Conv Embedding
#
#     """
#
#     def __init__(self,
#                  image_size,
#                  patch_size,
#                  kernel_size,
#                  batch_size,
#                  in_chans,
#                  embed_dim,
#                  stride,
#                  padding,
#                  norm_layer=None):
#         super().__init__()
#
#         self.proj = nn.Sequential(OrderedDict([
#             ('conv1', nn.Conv2d(
#                 in_chans, int(embed_dim),
#                 kernel_size=kernel_size,
#                 stride=stride,
#                 padding=padding,
#                 padding_mode='zeros',
#                 # groups=in_chans
#             )),
#             ('bn', nn.BatchNorm2d(int(embed_dim))),
#             ('relu', nn.GELU()),
#             ('pooling', nn.MaxPool2d(kernel_size=3, stride=stride, padding=1,)),
#             # ('pooling', nn.AdaptiveMaxPool2d((3, 3))),
#             # ('bn', nn.BatchNorm2d(int(embed_dim))),
#             # ('relu', nn.GELU()),
#         ]))
#
#     def forward(self, x):
#         sp_features = self.proj(x).to(device)  # proj_conv  proj
#
#         return sp_features


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout, image_size, patch_size, kernel_size, downsample, batch_size, in_chans,
                 patch_stride, patch_padding, norm_layer=nn.LayerNorm):
        super().__init__()

        # self.patch_embed = ConvEmbed(
        #     image_size=image_size,
        #     patch_size=patch_size,
        #     kernel_size=kernel_size,
        #     batch_size=batch_size,
        #     in_chans=in_chans,
        #     stride=patch_stride,
        #     padding=patch_padding,
        #     embed_dim=dim//4,
        #     norm_layer=norm_layer
        # )
        # self.patch_dim = (int(patch_size//4) ** 2) * int(dim) // 4
        self.dim = dim
        channels = 3
        self.patch_dim = channels * patch_size ** 2

        self.patch_to_embedding = nn.Linear(self.patch_dim, dim).to(device)

        self.layers = nn.ModuleList([])
        self.depth = depth

        for i in range(self.depth):
            self.layers.append(nn.ModuleList([
                # Residual(PreNorm(dim, Attention(depth, i, dim, image_size=image_size, patch_size=patch_size, heads=heads, dropout=dropout, downsample=downsample, kernel_size=kernel_size))),
                PreNorm(dim, Residual(Attention_3GIN(depth, i, dim, image_size=image_size, patch_size=patch_size, heads=heads, dropout=dropout, downsample=downsample, kernel_size=kernel_size))),
                FeedForward(dim, mlp_dim, image_size, patch_size, dropout=dropout),
            ]))

        self.dropout = nn.Dropout(dropout)

        # self.norm = nn.ModuleList([])
        # for _ in range(depth):
        #     self.norm.append(nn.LayerNorm(dim))

        self.patch_size = patch_size
        self.batch_size = batch_size
        self.head_num = heads

        self.Upool_out = nn.Sequential(
            nn.Linear(dim, 1, bias=False),
        )

    def forward(self, x, k):

        # p = self.patch_size
        # x = rearrange(img, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=p, p2=p)
        # x = self.patch_to_embedding(x)
        b, n, _ = x.shape

        for attn, ff in self.layers:
            # x = attn(x, self.rep_adj, 0)
            x, norm_attn_score = attn(x)
            x = ff(x)

        # norm_attn_score = self.sparse_adj(norm_attn_score, k)
        return x, norm_attn_score


class inv_vit(nn.Module):
    def __init__(self, *, image_size, patch_size, kernel_size, downsample, batch_size, num_classes, dim, depth, heads, mlp_dim, patch_stride, patch_pading, in_chans, dropout=0., emb_dropout = 0., expansion_factor=1):
        super().__init__()
        assert image_size % patch_size == 0, 'image dimensions must be divisible by the patch size'
        pantchesalow = image_size // patch_size
        num_patches = pantchesalow ** 2

        self.patch_size = patch_size

        # self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        # self.patch_to_embedding = nn.Linear(patch_dim, dim)
        # self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, mlp_dim, dropout, image_size, patch_size, kernel_size, downsample,
                                       batch_size, in_chans, patch_stride=patch_stride, patch_padding=patch_pading)

        self.to_cls_token = nn.Identity()
        self.heads = heads

        self.Upool = nn.Sequential(
            nn.Linear(dim, int(image_size / patch_size / 4) ** 2, bias=True),
        )

        self.proj_gate = nn.Sequential(
            nn.Linear(dim*2, dim, bias=True),
            nn.Sigmoid(),
        )

        self.new_size = (image_size//4, image_size//4)
        # self.unloader = transforms.ToPILImage()
        # self.mean = [0.485, 0.456, 0.406]
        # self.std = [0.229, 0.224, 0.225]
        # self.resize = transforms.Compose([
        #     transforms.Resize((image_size//4, image_size//4)),
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        #     ])

    def forward(self, x, k):

        # img = img.float()
        # images_denorm = img * torch.tensor(self.std).view(-1, 1, 1).to(device) + torch.tensor(self.mean).view(-1, 1,
        #                                                                                                       1).to(device)
        # images_denorm = images_denorm.type_as(img)
        # augmented_img1 = self.unloader(images_denorm[0, :, :, :]).convert('RGB')
        # img = self.resize(augmented_img1)
        # plt.imshow(augmented_img1, cmap="brg")
        # plt.show()

        x, norm_attn_score = self.transformer(x, k)

        batch_size = x.size(0)

        temp = self.Upool(x).permute(0, 2, 1)
        # C = F.gumbel_softmax(temp, dim=-1, tau=1.0)
        C = temp #  F.softmax(temp/2, dim=-1)
        x = torch.matmul(C, x)

        _, n, _ = x.size()

        C = C.unsqueeze(dim=1).expand(batch_size, self.heads, -1, -1).to(device)
        temp2 = torch.matmul(C, norm_attn_score)
        norm_attn_score = torch.matmul(temp2, C.permute(0, 1, 3, 2))  #好像也没什么用啊

        # x = self.to_cls_token(x[:, -1])
        return x, norm_attn_score