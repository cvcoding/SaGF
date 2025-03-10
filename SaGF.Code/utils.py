# -*- coding: utf-8 -*-

'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''
import os
import sys
import time
import math
from einops import rearrange
import torch.nn as nn
import torch.nn.init as init


def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)


try:
	_, term_width = os.popen('stty size', 'r').read().split()
except:
	term_width = 80
term_width = int(term_width)

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time
def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f



import cv2
import numpy as np

class ActivationsAndGradients:
    """ Class for extracting activations and
    registering gradients from targeted intermediate layers """

    def __init__(self, model, target_layers, reshape_transform):
        self.model = model
        self.gradients = []
        self.activations = []
        self.reshape_transform = reshape_transform
        self.handles = []
        for target_layer in target_layers:
            self.handles.append(
                target_layer.register_forward_hook(
                    self.save_activation))
            # Backward compatibility with older pytorch versions:
            #这里的if语句主要处理的是pytorch版本兼容问题
            if hasattr(target_layer, 'register_full_backward_hook'):
                self.handles.append(
                    target_layer.register_full_backward_hook(
                        self.save_gradient))
            else:
                self.handles.append(
                    target_layer.register_backward_hook(
                        self.save_gradient))
#正向传播由底层流向高层
    def save_activation(self, module, input, output):
        activation = output
        if self.reshape_transform is not None:
            activation = self.reshape_transform(activation)
        self.activations.append(activation.cpu().detach())
#反向传播有高层流向低层【保存的方式相反】
    def save_gradient(self, module, grad_input, grad_output):
        # Gradients are computed in reverse order
        grad = grad_output[0]
        if self.reshape_transform is not None:
            grad = self.reshape_transform(grad)
        self.gradients = [grad.cpu().detach()] + self.gradients
    #正向传播
    def __call__(self, x, gen_adj, label ):
        self.gradients = []
        self.activations = []
        return self.model(x, gen_adj, label, None, None)

    def release(self):
        for handle in self.handles:
            handle.remove()


class GradCAM:
    def __init__(self,
                 model,
                 target_layers,
                 reshape_transform=None,
                 #使用不同的设备，默认使用gpu
                 use_cuda=True):
        self.model = model.eval()#模型设置为验证模式
        self.target_layers = target_layers
        self.reshape_transform = reshape_transform
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()
        #ActivationsAndGradients正向传播获得的A和反向传播获得的A'
        self.activations_and_grads = ActivationsAndGradients(
            self.model, target_layers, reshape_transform)

    """ Get a vector of weights for every channel in the target layer.
        Methods that return weights channels,
        will typically need to only implement this function. """

    @staticmethod
    def get_cam_weights(grads):
        return np.mean(grads, axis=(2, 3), keepdims=True)

    @staticmethod
    def get_loss(output, target_category):
        loss = 0
        for i in range(len(target_category)):
            loss = loss + output[i, target_category[i]]#当前第i张图片获取的类别索引（感兴趣的类别值）
        return loss

    def get_cam_image(self, activations, grads):
        weights = self.get_cam_weights(grads)
        weighted_activations = weights * activations
        cam = weighted_activations.sum(axis=1)

        return cam

    @staticmethod
    def get_target_width_height(input_tensor):
        width, height = input_tensor.size(-1), input_tensor.size(-2)
        return width, height

    def compute_cam_per_layer(self, input_tensor):
        batch_size = input_tensor.size(0)
        patch_size = 15
        h = input_tensor.size(2)//patch_size
        activations_list = [a.cpu().data.numpy()
                            for a in self.activations_and_grads.activations]
        grads_list = [g.cpu().data.numpy()
                      for g in self.activations_and_grads.gradients]
        target_size = (patch_size, patch_size)  #self.get_target_width_height(input_tensor)

        cam_per_target_layer = []
        # Loop over the saliency image from every layer

        for layer_activations, layer_grads in zip(activations_list, grads_list):
            cam = self.get_cam_image(layer_activations, layer_grads)
            cam[cam < 0] = 0  # works like mute the min-max scale in the function of scale_cam_image，相当于使用了relu激活函数
            scaled = self.scale_cam_image(cam, target_size)
            cam_per_target_layer.append(scaled[:, None, :])
        cam_per_target_layer = rearrange(cam_per_target_layer[0], '(b h w) (c) (p1) (p2) -> b c (h p1) (w p2)', b=batch_size, h=h, w=h)

        return [cam_per_target_layer]

    def aggregate_multi_layers(self, cam_per_target_layer):
        cam_per_target_layer = np.concatenate(cam_per_target_layer, axis=1)
        cam_per_target_layer = np.maximum(cam_per_target_layer, 0)
        result = np.mean(cam_per_target_layer, axis=1)
        return self.scale_cam_image(result)

    @staticmethod
    def scale_cam_image(cam, target_size=None):
        result = []
        for img in cam:
            img = img - np.min(img)
            img = img / (1e-7 + np.max(img))
            if target_size is not None:
                img = cv2.resize(img, target_size)
            result.append(img)
        result = np.float32(result)

        return result

    def __call__(self, input_tensor, gen_adj, label, target_category=None):

        if self.cuda:
            input_tensor = input_tensor.cuda()

        # 正向传播得到网络输出logits(未经过softmax)
        output = self.activations_and_grads(input_tensor, gen_adj, label)[1]
        if isinstance(target_category, int):
            target_category = [target_category] * input_tensor.size(0)

        if target_category is None:
            target_category = np.argmax(output.cpu().data.numpy(), axis=-1)
            print(f"category id: {target_category}")
        else:
            assert (len(target_category) == input_tensor.size(0))

        self.model.zero_grad() # 清空历史梯度信息
        loss = self.get_loss(output, target_category)
        loss.backward(retain_graph=True) # 捕获对应的梯度信息并进行保存

        # In most of the saliency attribution papers, the saliency is
        # computed with a single target layer.
        # Commonly it is the last convolutional layer.
        # Here we support passing a list with multiple target layers.
        # It will compute the saliency image for every image,
        # and then aggregate them (with a default mean aggregation).
        # This gives you more flexibility in case you just want to
        # use all conv layers for example, all Batchnorm layers,
        # or something else.
        cam_per_layer = self.compute_cam_per_layer(input_tensor)
        return self.aggregate_multi_layers(cam_per_layer)

    def __del__(self):
        self.activations_and_grads.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.activations_and_grads.release()
        if isinstance(exc_value, IndexError):
            # Handle IndexError here...
            print(
                f"An exception occurred in CAM with block: {exc_type}. Message: {exc_value}")
            return True


def show_cam_on_image(img: np.ndarray,
                      mask: np.ndarray,
                      use_rgb: bool = False,
                      colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    """ This function overlays the cam mask on the image as an heatmap.
    By default the heatmap is in BGR format.

    :param img: The base image in RGB or BGR format.
    :param mask: The cam mask.
    :param use_rgb: Whether to use an RGB or BGR heatmap, this should be set to True if 'img' is in RGB format.
    :param colormap: The OpenCV colormap to be used.
    :returns: The default image with the cam overlay.
    """

    heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)
    if use_rgb:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255

    if np.max(img) > 1:
        raise Exception(
            "The input image should np.float32 in the range [0, 1]")

    cam = heatmap + img
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)


def center_crop_img(img: np.ndarray, size: int):
    h, w, c = img.shape

    if w == h == size:
        return img

    if w < h:
        ratio = size / w
        new_w = size
        new_h = int(h * ratio)
    else:
        ratio = size / h
        new_h = size
        new_w = int(w * ratio)

    img = cv2.resize(img, dsize=(new_w, new_h))

    if new_w == size:
        h = (new_h - size) // 2
        img = img[h: h+size]
    else:
        w = (new_w - size) // 2
        img = img[:, w: w+size]

    return img



# class ActivationsAndGradients:
#     """ Class for extracting activations and
#     registering gradients from targeted intermediate layers """
#     def __init__(self, model, target_layers, reshape_transform):
#         self.model = model
#         self.gradients = []
#         self.activations = []
#         self.reshape_transform = reshape_transform
#         self.handles = []
#         for target_layer in target_layers:
#             self.handles.append(
#                 target_layer.register_forward_hook(
#                     self.save_activation))
#             # Backward compatibility with older pytorch versions:
#             if hasattr(target_layer, 'register_full_backward_hook'):
#                 self.handles.append(
#                     target_layer.register_full_backward_hook(
#                         self.save_gradient))
#             else:
#                 self.handles.append(
#                     target_layer.register_backward_hook(
#                         self.save_gradient))
#     def save_activation(self, module, input, output):
#         activation = output
#         if self.reshape_transform is not None:
#             activation = self.reshape_transform(activation)
#         self.activations.append(activation.cpu().detach())
#     def save_gradient(self, module, grad_input, grad_output):
#         # Gradients are computed in reverse order
#         grad = grad_output[0]
#         if self.reshape_transform is not None:
#             grad = self.reshape_transform(grad)
#         self.gradients = [grad.cpu().detach()] + self.gradients
#     def __call__(self, x, gen_adj, label):
#         self.gradients = []
#         self.activations = []
#         return self.model(x, gen_adj, label, None, None)
#     def release(self):
#         for handle in self.handles:
#             handle.remove()
# class GradCAM:
#     def __init__(self,
#                  model,
#                  target_layers,
#                  reshape_transform=None,
#                  use_cuda=False):
#         self.model = model.eval()
#         self.target_layers = target_layers
#         self.reshape_transform = reshape_transform
#         self.cuda = use_cuda
#         if self.cuda:
#             self.model = model.cuda()
#         self.activations_and_grads = ActivationsAndGradients(
#             self.model, target_layers, reshape_transform)
#     """ Get a vector of weights for every channel in the target layer.
#         Methods that return weights channels,
#         will typically need to only implement this function. """
#     @staticmethod
#     def get_cam_weights(grads):
#         return np.mean(grads, axis=(2, 3), keepdims=True)
#     @staticmethod
#     def get_loss(output, target_category):
#         loss = 0
#         for i in range(len(target_category)):
#             loss = loss + output[i, target_category[i]]
#         return loss
#     def get_cam_image(self, activations, grads):
#         weights = self.get_cam_weights(grads)
#         weighted_activations = weights * activations
#         cam = weighted_activations.sum(axis=1)
#         return cam
#     @staticmethod
#     def get_target_width_height(input_tensor):
#         width, height = input_tensor.size(-1), input_tensor.size(-2)
#         return width, height
#     def compute_cam_per_layer(self, input_tensor):
#         activations_list = [a.cpu().data.numpy()
#                             for a in self.activations_and_grads.activations]
#         grads_list = [g.cpu().data.numpy()
#                       for g in self.activations_and_grads.gradients]
#         target_size = self.get_target_width_height(input_tensor)
#         cam_per_target_layer = []
#         # Loop over the saliency image from every layer
#         for layer_activations, layer_grads in zip(activations_list, grads_list):
#             cam = self.get_cam_image(layer_activations, layer_grads)
#             cam[cam < 0] = 0  # works like mute the min-max scale in the function of scale_cam_image
#             scaled = self.scale_cam_image(cam, target_size)
#             cam_per_target_layer.append(scaled[:, None, :])
#         return cam_per_target_layer
#     def aggregate_multi_layers(self, cam_per_target_layer):
#         cam_per_target_layer = np.concatenate(cam_per_target_layer, axis=1)
#         cam_per_target_layer = np.maximum(cam_per_target_layer, 0)
#         result = np.mean(cam_per_target_layer, axis=1)
#         return self.scale_cam_image(result)
#     @staticmethod
#     def scale_cam_image(cam, target_size=None):
#         result = []
#         for img in cam:
#             img = img
#             img = img - np.min(img)
#             img = img / (1e-7 + np.max(img))
#             if target_size is not None:
#                 img = cv2.resize(img, target_size)
#             result.append(img)
#         result = np.float32(result)
#         return result
#     def __call__(self, input_tensor, gen_adj, label, target_category=None):
#         if self.cuda:
#             input_tensor = input_tensor.cuda()
#         # 正向传播得到网络输出logits(未经过softmax)
#         output = self.activations_and_grads(input_tensor, gen_adj, label)[1]
#         if isinstance(target_category, int):
#             target_category = [target_category] * input_tensor.size(0)
#         if target_category is None:
#             target_category = np.argmax(output.cpu().data.numpy(), axis=-1)
#             print(f"category id: {target_category}")
#         else:
#             assert (len(target_category) == input_tensor.size(0))
#         self.model.zero_grad()
#         loss = self.get_loss(output, target_category)
#         loss.backward(retain_graph=True)
#         # In most of the saliency attribution papers, the saliency is
#         # computed with a single target layer.
#         # Commonly it is the last convolutional layer.
#         # Here we support passing a list with multiple target layers.
#         # It will compute the saliency image for every image,
#         # and then aggregate them (with a default mean aggregation).
#         # This gives you more flexibility in case you just want to
#         # use all conv layers for example, all Batchnorm layers,
#         # or something else.
#         cam_per_layer = self.compute_cam_per_layer(input_tensor)
#         return self.aggregate_multi_layers(cam_per_layer)
#     def __del__(self):
#         self.activations_and_grads.release()
#     def __enter__(self):
#         return self
#     def __exit__(self, exc_type, exc_value, exc_tb):
#         self.activations_and_grads.release()
#         if isinstance(exc_value, IndexError):
#             # Handle IndexError here...
#             print(
#                 f"An exception occurred in CAM with block: {exc_type}. Message: {exc_value}")
#             return True
# def show_cam_on_image(img: np.ndarray,
#                       mask: np.ndarray,
#                       use_rgb: bool = False,
#                       colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
#     """ This function overlays the cam mask on the image as an heatmap.
#     By default the heatmap is in BGR format.
#     :param img: The base image in RGB or BGR format.
#     :param mask: The cam mask.
#     :param use_rgb: Whether to use an RGB or BGR heatmap, this should be set to True if 'img' is in RGB format.
#     :param colormap: The OpenCV colormap to be used.
#     :returns: The default image with the cam overlay.
#     """
#     heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)
#     if use_rgb:
#         heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
#     heatmap = np.float32(heatmap) / 255
#     if np.max(img) > 1:
#         raise Exception(
#             "The input image should np.float32 in the range [0, 1]")
#     cam = heatmap + img
#     cam = cam / np.max(cam)
#     return np.uint8(255 * cam)
# def center_crop_img(img: np.ndarray, size: int):
#     h, w, c = img.shape
#     if w == h == size:
#         return img
#     if w < h:
#         ratio = size / w
#         new_w = size
#         new_h = int(h * ratio)
#     else:
#         ratio = size / h
#         new_h = size
#         new_w = int(w * ratio)
#     img = cv2.resize(img, dsize=(new_w, new_h))
#     if new_w == size:
#         h = (new_h - size) // 2
#         img = img[h: h+size]
#     else:
#         w = (new_w - size) // 2
#         img = img[:, w: w+size]
#     return img



# Copyright 2020 - 2022 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import torch
from numpy.random import randint


def patch_rand_drop(x, x_rep=None, max_drop=0.5, max_block_sz=0.25, tolr=0.05):
    c, h, w = x.size()
    n_drop_pix = np.random.uniform(0.1, max_drop) * h * w
    mx_blk_height = int(h * max_block_sz)
    mx_blk_width = int(w * max_block_sz)
    tolr = (int(tolr * h), int(tolr * w))
    total_pix = 0
    while total_pix < n_drop_pix:
        rnd_r = randint(0, h - tolr[0])
        rnd_c = randint(0, w - tolr[1])
        rnd_h = min(randint(tolr[0], mx_blk_height) + rnd_r, h)
        rnd_w = min(randint(tolr[1], mx_blk_width) + rnd_c, w)
        if x_rep is None:
            # x_uninitialized = torch.empty(
            #     (c, rnd_h - rnd_r, rnd_w - rnd_c), dtype=x.dtype, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            # ).normal_()
            # x_uninitialized = (x_uninitialized - torch.min(x_uninitialized)) / (
            #     torch.max(x_uninitialized) - torch.min(x_uninitialized)
            # )
            mean_val = x.mean(dim=(1, 2), keepdim=True)  # x_uninitialized  #
            x[:, rnd_r:rnd_h, rnd_c:rnd_w] = mean_val
        else:
            x[:, rnd_r:rnd_h, rnd_c:rnd_w] = x_rep[:, rnd_r:rnd_h, rnd_c:rnd_w]
        total_pix = total_pix + (rnd_h - rnd_r) * (rnd_w - rnd_c)
    return x


def rot_rand(x_s):
    img_n = x_s.size()[0]
    x_aug = x_s.detach().clone()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    x_rot = torch.zeros(img_n).long().to(device)
    for i in range(img_n):
        x = x_s[i]
        orientation = np.random.randint(0, 4)
        if orientation == 0:
            pass
        elif orientation == 1:
            x = x.rot90(1, (2, 3))
        elif orientation == 2:
            x = x.rot90(2, (2, 3))
        elif orientation == 3:
            x = x.rot90(3, (2, 3))
        x_aug[i] = x
        x_rot[i] = orientation
    return x_aug, x_rot


def aug_rand(samples):
    x_aug = samples.detach().clone()
    x_aug = patch_rand_drop(x_aug)
    return x_aug