import copy
import random
from functools import wraps
import math
import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from torchvision import transforms as T

# helper functions


def default(val, def_val):
    return def_val if val is None else val


def flatten(t):
    return t.reshape(t.shape[0], -1)


def singleton(cache_key):
    def inner_fn(fn):
        @wraps(fn)
        def wrapper(self, *args, **kwargs):
            instance = getattr(self, cache_key)
            if instance is not None:
                return instance

            instance = fn(self, *args, **kwargs)
            setattr(self, cache_key, instance)
            return instance
        return wrapper
    return inner_fn


def get_module_device(module):
    return next(module.parameters()).device


def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val


def MaybeSyncBatchnorm(is_distributed = None):
    is_distributed = default(is_distributed, dist.is_initialized() and dist.get_world_size() > 1)
    return nn.SyncBatchNorm if is_distributed else nn.BatchNorm1d

# loss fn


def loss_fn(x, y):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return 2 - 2 * (x * y).sum(dim=-1)

mse_loss = nn.MSELoss()
L1_loss = nn.L1Loss()
# augmentation utils


class RandomApply(nn.Module):
    def __init__(self, fn, p):
        super().__init__()
        self.fn = fn
        self.p = p
    def forward(self, x):
        if random.random() > self.p:
            return x
        return self.fn(x)

# exponential moving average


class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


def update_moving_average(ema_updater, ma_model, current_model):
    for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = ema_updater.update_average(old_weight, up_weight)

# MLP class for projector and predictor


def MLP(dim, projection_size, hidden_size=4096, sync_batchnorm=None):
    return nn.Sequential(
        nn.Linear(dim, hidden_size),
        MaybeSyncBatchnorm(sync_batchnorm)(hidden_size),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_size, projection_size)
    )


def SimSiamMLP(dim, projection_size, hidden_size=4096, sync_batchnorm=None):
    return nn.Sequential(
        nn.Linear(dim, hidden_size, bias=False),
        MaybeSyncBatchnorm(sync_batchnorm)(hidden_size),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_size, hidden_size, bias=False),
        MaybeSyncBatchnorm(sync_batchnorm)(hidden_size),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_size, projection_size, bias=False),
        MaybeSyncBatchnorm(sync_batchnorm)(projection_size, affine=False)
    )

# a wrapper class for the base neural network
# will manage the interception of the hidden layer output
# and pipe it into the projecter and predictor nets


class NetWrapper(nn.Module):
    def __init__(self, net, projection_size, projection_hidden_size, layer=-2, use_simsiam_mlp = False, sync_batchnorm = None):
        super().__init__()
        self.net = net
        self.layer = layer

        self.projector = None
        self.projection_size = projection_size
        self.projection_hidden_size = projection_hidden_size

        self.use_simsiam_mlp = use_simsiam_mlp
        self.sync_batchnorm = sync_batchnorm

        self.hidden = {}
        self.hook_registered = False

    def _find_layer(self):
        if type(self.layer) == str:
            modules = dict([*self.net.named_modules()])
            return modules.get(self.layer, None)
        elif type(self.layer) == int:
            children = [*self.net.children()]
            return children[self.layer]
        return None

    def _hook(self, _, input, output):
        device = input[0].device
        self.hidden[device] = flatten(output)

    def _register_hook(self):
        layer = self._find_layer()
        assert layer is not None, f'hidden layer ({self.layer}) not found'
        handle = layer.register_forward_hook(self._hook)
        self.hook_registered = True

    @singleton('projector')
    def _get_projector(self, hidden):
        _, dim = hidden.shape
        create_mlp_fn = MLP if not self.use_simsiam_mlp else SimSiamMLP
        projector = create_mlp_fn(dim, self.projection_size, self.projection_hidden_size, sync_batchnorm=self.sync_batchnorm)
        return projector.to(hidden)

    def get_representation(self, x, t_x, x_lowrank, noise, gen_adj, rank_k):
        return self.net(x, t_x, x_lowrank, noise, gen_adj, rank_k)

        # if self.layer == -1:
        #     return self.net(x)
        #
        # if not self.hook_registered:
        #     self._register_hook()
        #
        # self.hidden.clear()
        # _ = self.net(x)
        # hidden = self.hidden[x.device]
        # self.hidden.clear()
        #
        # assert hidden is not None, f'hidden layer {self.layer} never emitted an output'
        # return hidden

    def forward(self, x, t_x, x_lowrank, noise, gen_adj, rank_k, return_projection=True):
        representation, _ = self.get_representation(x, t_x, x_lowrank, noise, gen_adj, rank_k)

        if not return_projection:
            return representation

        projector = self._get_projector(representation)
        projection = projector(representation)
        return projection, representation


# main class
class BYOL(nn.Module):
    def __init__(
        self,
        net,
        gen_vit,
        # inv_vit,
        image_size,
        hidden_layer,
        projection_size,
        projection_hidden_size,
        augment_fn=None,
        augment_fn2=None,
        moving_average_decay=0.99,
        use_momentum=True,
        sync_batchnorm=None
    ):
        super().__init__()
        self.net = net

        # default SimCLR augmentation

        DEFAULT_AUG = torch.nn.Sequential(
            # T.Scale(int(1.2*image_size)),
            # RandomApply(
            #     T.ColorJitter(0.8, 0.8, 0.8, 0.2),
            #     p = 0.3
            # ),
            # T.RandomGrayscale(p=0.2),
            # T.RandomHorizontalFlip(),
            # RandomApply(
            #     T.GaussianBlur((3, 3), (1.0, 2.0)),
            #     p = 0.2
            # ),
            T.RandomRotation(degrees=30),  #, fill=(255, 255, 255)
            # T.RandomResizedCrop(size=image_size, scale=(0.6, 0.9)),  #, ratio=(0.99, 1.0)
            # T.RandomResizedCrop((image_size, image_size)),
            # T.Normalize(
            #     mean=torch.tensor([0.485, 0.456, 0.406]),
            #     std=torch.tensor([0.229, 0.224, 0.225])),
        )

        self.augment1 = default(augment_fn, DEFAULT_AUG)
        self.augment2 = default(augment_fn2, self.augment1)

        self.online_encoder = NetWrapper(
            net,
            projection_size,
            projection_hidden_size,
            layer=hidden_layer,
            use_simsiam_mlp=not use_momentum,
            sync_batchnorm=sync_batchnorm
        )
        self.gen_vit = gen_vit
        # self.inv_vit = inv_vit
        self.use_momentum = use_momentum
        self.target_encoder = None
        self.target_gen_vit = None

        self.target_ema_updater = EMA(moving_average_decay)

        self.online_predictor = MLP(projection_size, projection_size, projection_hidden_size)

        # get device of network and make wrapper same device
        self.device = get_module_device(net)
        self.to(self.device)

        pantchesalow = image_size // self.net.patch_size // 2
        num_patches = pantchesalow ** 2
        # self.k = int(math.sqrt(num_patches))
        self.A1 = nn.Parameter(torch.randn(num_patches, num_patches) * 0.01).to(self.device)
        self.A2 = nn.Parameter(torch.randn(num_patches, num_patches) * 0.01).to(self.device)
        self.m = nn.Parameter(torch.randn(num_patches)).to(self.device)

        # random_matrix = torch.rand(*adj_matrix.shape)
        # adj_matrix[random_matrix < 0.8] = 1

        # send a mock image tensor to instantiate singleton parameters
        # self.forward(torch.randn(2, 3, image_size, image_size, device=device), torch.randn(2, 3, image_size, image_size, device=device))

    @singleton('target_encoder')
    def _get_target_encoder(self):
        target_encoder = copy.deepcopy(self.online_encoder)
        set_requires_grad(target_encoder, False)
        return target_encoder

    @singleton('target_gen_vit')
    def _get_target_gen_vit(self):
        target_gen_vit = copy.deepcopy(self.gen_vit)
        set_requires_grad(target_gen_vit, False)
        return target_gen_vit

    def reset_moving_average(self):
        del self.target_encoder
        self.target_encoder = None

    def update_moving_average(self):
        assert self.use_momentum, 'you do not need to update the moving average, since you have turned off momentum for the target encoder'
        assert self.target_encoder is not None, 'target encoder has not been created yet'
        update_moving_average(self.target_ema_updater, self.target_encoder, self.online_encoder)
        update_moving_average(self.target_ema_updater, self.target_gen_vit, self.gen_vit)

    def forward(
        self,
        img1, img2, adj_matrix, rank_k,
        return_embedding=False,
        return_projection=True
    ):
        assert not (self.training and img1.shape[0] == 1), 'you must have greater than 1 sample when training, due to the batchnorm in the projection layer'

        if return_embedding:
            return self.online_encoder(img1, return_projection=return_projection)

        image_one, image_two = self.augment1(img1), self.augment2(img2)

        x_pre1, x1, noise1, x_sparsenoise_pool1, gen_adj_one, sparsity_v1 = self.gen_vit(image_one)
        x_pre2, x2, noise2, x_sparsenoise_pool2, gen_adj_two, sparsity_v2 = self.gen_vit(image_two)

        top_values, top_indices = torch.topk(self.m, rank_k)
        # 创建一个与原向量相同形状的零向量
        zeroed_vector = torch.zeros_like(self.m).cuda()
        # 将前 k 个最大值放回对应的位置
        zeroed_vector.scatter_(0, top_indices, top_values)
        fuzhuM = self.A1 @ torch.diag(zeroed_vector) @ self.A2
        loss_lowrank1 = mse_loss(x1, fuzhuM @ x1)
        loss_lowrank2 = mse_loss(x2, fuzhuM @ x2)

        # x_inv1, attn_inv1 = self.inv_vit(x1 + noise1, rank_k)
        # x_inv2, attn_inv2 = self.inv_vit(x2 + noise2, rank_k)

        # loss_reconstructed1 = mse_loss(x_pre1, x_inv1)
        # loss_reconstructed2 = mse_loss(x_pre2, x_inv2)

        # top_values, top_indices = torch.topk(self.m, rank_k)
        # # 创建一个与原向量相同形状的零向量
        # zeroed_vector = torch.zeros_like(self.m).cuda()
        # # 将前 k 个最大值放回对应的位置
        # zeroed_vector.scatter_(0, top_indices, top_values)
        # fuzhuM = self.A1 @ torch.diag(zeroed_vector) @ self.A2
        # loss_lowrank1 = mse_loss(x1, fuzhuM @ x1)
        # loss_lowrank2 = mse_loss(x2, fuzhuM @ x2)

        loss_sparse1 = L1_loss(noise1, torch.tensor([0.0]).expand_as(noise1).to(self.device))
        loss_sparse2 = L1_loss(noise2, torch.tensor([0.0]).expand_as(noise2).to(self.device))

        b, _, _, _ = image_one.shape

        online_proj_one, _ = self.online_encoder(image_one, x_pre1, x1, x_sparsenoise_pool1, gen_adj_one, sparsity_v1)
        online_proj_two, _ = self.online_encoder(image_two, x_pre2, x2, x_sparsenoise_pool1, gen_adj_two, sparsity_v2)
        # label ==0: adj_matrix; label==1: gen_adj
        online_pred_one = self.online_predictor(online_proj_one)
        online_pred_two = self.online_predictor(online_proj_two)

        with torch.no_grad():
            target_gen_vit = self._get_target_gen_vit() if self.use_momentum else self.gen_vit
            t_pre1, t_x1, t_noise1, t_x_sparsenoise_pool1, target_gen_adj_one, t_sparsity_v1 = target_gen_vit(image_one)

            t_pre2, t_x2, t_noise2, t_x_sparsenoise_pool2, target_gen_adj_two, t_sparsity_v2 = target_gen_vit(image_two)
            target_gen_adj_two = target_gen_adj_two.detach()
            target_gen_adj_one = target_gen_adj_one.detach()

            target_encoder = self._get_target_encoder() if self.use_momentum else self.online_encoder
            target_proj_one, _ = target_encoder(image_one, t_pre1, t_x1, t_x_sparsenoise_pool1, target_gen_adj_one.detach(), t_sparsity_v1)
            target_proj_two, _ = target_encoder(image_two, t_pre2, t_x2, t_x_sparsenoise_pool2, target_gen_adj_two.detach(), t_sparsity_v2)
            target_proj_one.detach_()
            target_proj_two.detach_()

        loss_one = loss_fn(online_pred_one, target_proj_two.detach())
        loss_two = loss_fn(online_pred_two, target_proj_one.detach())

        loss_onetwo = (loss_one + loss_two).mean()

        loss = loss_onetwo + loss_lowrank1 + loss_lowrank2 + 1.0*(loss_sparse1 + loss_sparse2)  # + loss_reconstructed1 + loss_reconstructed2
        return loss
