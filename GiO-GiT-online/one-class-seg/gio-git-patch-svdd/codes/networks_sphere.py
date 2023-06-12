# ------------------------------------------------------------------------------
# Modified from 'Anomaly-Detection-PatchSVDD-PyTorch' 
# Reference: https://github.com/nuclearboy95/Anomaly-Detection-PatchSVDD-PyTorch
# ------------------------------------------------------------------------------
import os
import math
import numpy as np
import scipy.io as io

import torch.nn as nn
import torch
import torch.nn.functional as F

import matplotlib 
import matplotlib.pyplot as plt
plt.switch_backend('agg')

from .utils import makedirpath
from torchsphere import nn as spherenn
from torchsphere.nn import AngleLoss

__all__ = ['EncoderHier', 'Encoder', 'PositionClassifier']
xent = nn.CrossEntropyLoss()
angleloss = AngleLoss()


class Encoder(nn.Module):
    def __init__(self, K, D=64, bias=True):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 64, 5, 2, 0, bias=bias)
        self.conv2 = nn.Conv2d(64, 64, 5, 2, 0, bias=bias)
        self.conv3 = nn.Conv2d(64, 128, 5, 2, 0, bias=bias)
        self.conv4 = nn.Conv2d(128, D, 5, 1, 0, bias=bias)

        self.K = K
        self.D = D
        self.bias = bias

    def forward(self, x):
        h = self.conv1(x)
        h = F.leaky_relu(h, 0.1)

        h = self.conv2(h)
        h = F.leaky_relu(h, 0.1)

        h = self.conv3(h)

        if self.K == 64:
            h = F.leaky_relu(h, 0.1)
            h = self.conv4(h)

        h = torch.tanh(h)

        return h

    def save(self, name):
        fpath = self.fpath_from_name(name)
        makedirpath(fpath)
        torch.save(self.state_dict(), fpath)

    def load(self, name):
        fpath = self.fpath_from_name(name)
        self.load_state_dict(torch.load(fpath))

    @staticmethod
    def fpath_from_name(name):
        return f'ckpts_sphere/{name}/encoder_nohier.pkl'


def forward_hier(x, emb_small, K):
    K_2 = K // 2
    n = x.size(0)
    x1 = x[..., :K_2, :K_2]
    x2 = x[..., :K_2, K_2:]
    x3 = x[..., K_2:, :K_2]
    x4 = x[..., K_2:, K_2:]
    xx = torch.cat([x1, x2, x3, x4], dim=0)
    hh = emb_small(xx)

    h1 = hh[:n]
    h2 = hh[n: 2 * n]
    h3 = hh[2 * n: 3 * n]
    h4 = hh[3 * n:]

    h12 = torch.cat([h1, h2], dim=3)
    h34 = torch.cat([h3, h4], dim=3)
    h = torch.cat([h12, h34], dim=2)
    return h


class EncoderDeep(nn.Module):
    def __init__(self, K, D=64, bias=True):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, 3, 2, 0, bias=bias)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, 0, bias=bias)
        self.conv3 = nn.Conv2d(64, 128, 3, 1, 0, bias=bias)
        self.conv4 = nn.Conv2d(128, 128, 3, 1, 0, bias=bias)
        self.conv5 = nn.Conv2d(128, 64, 3, 1, 0, bias=bias)
        self.conv6 = nn.Conv2d(64, 32, 3, 1, 0, bias=bias)
        self.conv7 = nn.Conv2d(32, 32, 3, 1, 0, bias=bias)
        self.conv8 = nn.Conv2d(32, D, 3, 1, 0, bias=bias)

        self.K = K
        self.D = D

    def forward(self, x):
        h = self.conv1(x)
        h = F.leaky_relu(h, 0.1)

        h = self.conv2(h)
        h = F.leaky_relu(h, 0.1)

        h = self.conv3(h)
        h = F.leaky_relu(h, 0.1)

        h = self.conv4(h)
        h = F.leaky_relu(h, 0.1)

        h = self.conv5(h)
        h = F.leaky_relu(h, 0.1)

        h = self.conv6(h)
        h = F.leaky_relu(h, 0.1)

        h = self.conv7(h)
        h = F.leaky_relu(h, 0.1)

        h = self.conv8(h)
        h = torch.tanh(h)

        return h

    def save(self, name):
        fpath = self.fpath_from_name(name)
        makedirpath(fpath)
        torch.save(self.state_dict(), fpath)

    def load(self, name):
        fpath = self.fpath_from_name(name)
        self.load_state_dict(torch.load(fpath))

    @staticmethod
    def fpath_from_name(name):
        return f'ckpts_sphere/{name}/encdeep.pkl'


class NormalizedLinear(nn.Module):
    __constants__ = ['bias', 'in_features', 'out_features']

    def __init__(self, in_features, out_features, bias=True):
        super(NormalizedLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        with torch.no_grad():
            w = self.weight / self.weight.data.norm(keepdim=True, dim=0)
        return F.linear(x, w, self.bias)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


class EncoderHier(nn.Module):
    def __init__(self, K, D=64, D_emb=128, bias=True):
        super().__init__()

        if K > 64:
            self.enc = EncoderHier(K // 2, D, bias=bias)

        elif K == 64:
            self.enc = EncoderDeep(K // 2, D, bias=bias)

        else:
            raise ValueError()

        self.conv1 = nn.Conv2d(D, D_emb, 2, 1, 0, bias=bias)
        self.conv2 = nn.Conv2d(D_emb, D, 1, 1, 0, bias=bias)

        self.K = K
        self.D = D

    def forward(self, x):
        h = forward_hier(x, self.enc, K=self.K)

        h = self.conv1(h)
        h = F.leaky_relu(h, 0.1)

        h = self.conv2(h)
        h = torch.tanh(h)

        return h

    def save(self, name):
        fpath = self.fpath_from_name(name)
        makedirpath(fpath)
        torch.save(self.state_dict(), fpath)

    def save_curve(self, name, 
                   curve_seg_64,
                   curve_seg_32,
                   curve_seg_sum,
                   curve_seg_mult):

        fpath = f'ckpts_sphere/{name}'
        makedirpath(fpath)

        plt.title('Seg_64')
        plt.plot(np.arange(len(curve_seg_64)), curve_seg_64)
        plt.ylim(0, 100)
        plt.xlabel('Epoch')
        plt.ylabel('AUROC (%)')
        plt.savefig(os.path.join(fpath, 'seg_64.jpg'))
        plt.close()

        plt.title('Seg_32')
        plt.plot(np.arange(len(curve_seg_32)), curve_seg_32)
        plt.ylim(0, 100)
        plt.xlabel('Epoch')
        plt.ylabel('AUROC (%)')
        plt.savefig(os.path.join(fpath, 'seg_32.jpg'))
        plt.close()

        plt.title('Seg_sum')
        plt.plot(np.arange(len(curve_seg_sum)), curve_seg_sum)
        plt.ylim(0, 100)
        plt.xlabel('Epoch')
        plt.ylabel('AUROC (%)')
        plt.savefig(os.path.join(fpath, 'seg_sum.jpg'))
        plt.close()

        plt.title('Seg_mult')
        plt.plot(np.arange(len(curve_seg_mult)), curve_seg_mult)
        plt.ylim(0, 100)
        plt.xlabel('Epoch')
        plt.ylabel('AUROC (%)')
        plt.savefig(os.path.join(fpath, 'seg_mult.jpg'))
        plt.close()

    def load(self, name):
        fpath = self.fpath_from_name(name)
        self.load_state_dict(torch.load(fpath))

    @staticmethod
    def fpath_from_name(name):
        return f'ckpts_sphere/{name}/enchier.pkl'


class PositionClassifier(nn.Module):
    def __init__(self, K, D, D_emb=128, class_num=8, curvature=1.0):
        super().__init__()
        self.D = D

        # define linear layers
        self.fc1  = nn.Linear(D, D_emb)
        self.act1 = nn.LeakyReLU(0.1)
        self.fc2  = nn.Linear(D_emb, D_emb)
        self.act2 = nn.LeakyReLU(0.1)
        self.fc3  = NormalizedLinear(D_emb, class_num)

        # define spherical layers
        self.fc1_s  = nn.Linear(D, D_emb)
        self.fc2_s  = nn.Linear(D_emb, D_emb)
        self.fc3_s  = spherenn.AngleLinear(D_emb, class_num, curvature=curvature, m=1)

        self.K = K

    def save(self, name):
        fpath = self.fpath_from_name(name)
        makedirpath(fpath)
        torch.save(self.state_dict(), fpath)

    def load(self, name):
        fpath = self.fpath_from_name(name)
        self.load_state_dict(torch.load(fpath))

    def fpath_from_name(self, name):
        return f'ckpts_sphere/{name}/position_classifier_K{self.K}.pkl'

    @staticmethod
    def infer(c, enc, batch):
        x1s, x2s, ys = batch

        x1 = enc(x1s)
        x2 = enc(x2s)

        logits, logits_s, _, _, _, _ = c(x1, x2)

        loss1 = xent(logits, ys)
        loss2 = angleloss(logits_s, ys)

        return loss1 + loss2

    def forward(self, x1, x2):
        x1 = x1.view(-1, self.D)
        x2 = x2.view(-1, self.D)
        x = x1 - x2

        # spherical layer
        x_s = self.fc1_s(x)
        info_s1 = x_s
        x_s = self.act1(x_s)
        x_s = self.fc2_s(x_s)
        info_s2 = x_s
        x_s = self.act2(x_s)
        x_s = self.fc3_s(x_s)

        # linear layer
        x = self.fc1(x)
        info_l1 = x
        x = self.act1(x)
        x = self.fc2(x)
        info_l2 = x
        x = self.act2(x)
        x = self.fc3(x)

        return x, x_s, info_l1, info_s1, info_l2, info_s2
