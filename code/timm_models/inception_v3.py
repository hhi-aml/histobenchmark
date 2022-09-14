""" Inception-V3
Originally from torchvision Inception3 model
Licensed BSD-Clause 3 https://github.com/pytorch/vision/blob/master/LICENSE
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from pytorch_LRP.layers import *

from timm.data import IMAGENET_DEFAULT_STD, IMAGENET_DEFAULT_MEAN, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
# TODO:
from timm.models.helpers import build_model_with_cfg, resolve_pretrained_cfg, flatten_modules
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_

#from .layers import trunc_normal_, create_classifier, Linear


class Linear2(Linear):
    r"""Applies a linear transformation to the incoming data: :math:`y = xA^T + b`
    Wraps torch.nn.Linear to support AMP + torchscript usage by manually casting
    weight & bias to input.dtype to work around an issue w/ torch.addmm in this use case.
    """
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if torch.jit.is_scripting():
            bias = self.bias.to(dtype=input.dtype) if self.bias is not None else None
            return F.linear(input, self.weight.to(dtype=input.dtype), bias=bias)
        else:
            return F.linear(input, self.weight, self.bias)


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 299, 299), 'pool_size': (8, 8),
        'crop_pct': 0.875, 'interpolation': 'bicubic',
        'mean': IMAGENET_INCEPTION_MEAN, 'std': IMAGENET_INCEPTION_STD,
        'first_conv': 'Conv2d_1a_3x3.conv', 'classifier': 'fc',
        **kwargs
    }


default_cfgs = {
    # original PyTorch weights, ported from Tensorflow but modified
    'inception_v3': _cfg(
        url='https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth',
        has_aux=True),  # checkpoint has aux logit layer weights
    # my port of Tensorflow SLIM weights (http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz)
    'tf_inception_v3': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_inception_v3-e0069de4.pth',
        num_classes=1000, has_aux=False, label_offset=1),
    # my port of Tensorflow adversarially trained Inception V3 from
    # http://download.tensorflow.org/models/adv_inception_v3_2017_08_18.tar.gz
    'adv_inception_v3': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/adv_inception_v3-9e27bd63.pth',
        num_classes=1000, has_aux=False, label_offset=1),
    # from gluon pretrained models, best performing in terms of accuracy/loss metrics
    # https://gluon-cv.mxnet.io/model_zoo/classification.html
    'gluon_inception_v3': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/gluon_inception_v3-9f746940.pth',
        mean=IMAGENET_DEFAULT_MEAN,  # also works well with inception defaults
        std=IMAGENET_DEFAULT_STD,  # also works well with inception defaults
        has_aux=False,
    )
}


class InceptionA(nn.Module):

    def __init__(self, in_channels, pool_features, conv_block=None):
        super(InceptionA, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.branch1x1 = conv_block(in_channels, 64, kernel_size=1)

        self.branch5x5_1 = conv_block(in_channels, 48, kernel_size=1)
        self.branch5x5_2 = conv_block(48, 64, kernel_size=5, padding=2)

        self.branch3x3dbl_1 = conv_block(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = conv_block(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = conv_block(96, 96, kernel_size=3, padding=1)

        self.branch_pool = conv_block(in_channels, pool_features, kernel_size=1)
        self.avg_pool = AvgPool2d(kernel_size=3, stride=1, padding=1)

    def _forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = self.avg_pool(x)#F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        self.branch_lens = np.cumsum([b.shape[1] for b in outputs])
        return outputs

    def forward(self, x):
        outputs = self._forward(x)
        return torch.cat(outputs, 1)

    def relprop(self, R, alpha=2/3):
        # undo concatenation
        bl = [0] + self.branch_lens.tolist()
        branch1x1, branch5x5, branch3x3dbl, branch_pool = [R[:, bl[i]:bl[i+1]] for i in range(len(bl)-1)]
        # relevance of the pooling branch
        branch_pool = self.branch_pool.relprop(branch_pool, alpha)
        branch_pool = self.avg_pool.relprop(branch_pool, alpha)
        # relevance of the 3x3 branch
        branch3x3dbl = self.branch3x3dbl_3.relprop(branch3x3dbl, alpha)
        branch3x3dbl = self.branch3x3dbl_2.relprop(branch3x3dbl, alpha)
        branch3x3dbl = self.branch3x3dbl_1.relprop(branch3x3dbl, alpha)
        # relevance of the 5x5 branch
        branch5x5 = self.branch5x5_2.relprop(branch5x5, alpha)
        branch5x5 = self.branch5x5_1.relprop(branch5x5, alpha)
        # relevance of the 1x1 branch
        branch1x1 = self.branch1x1.relprop(branch1x1, alpha)
        # merge branch relevances by adding them up
        return branch1x1 + branch5x5 + branch3x3dbl + branch_pool


class InceptionB(nn.Module):

    def __init__(self, in_channels, conv_block=None):
        super(InceptionB, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.branch3x3 = conv_block(in_channels, 384, kernel_size=3, stride=2)

        self.branch3x3dbl_1 = conv_block(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = conv_block(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = conv_block(96, 96, kernel_size=3, stride=2)

        self.max_pool = MaxPool2d(kernel_size=3, stride=2)

    def _forward(self, x):
        branch3x3 = self.branch3x3(x)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        #branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)
        branch_pool = self.max_pool(x)#, kernel_size=3, stride=2)

        outputs = [branch3x3, branch3x3dbl, branch_pool]
        self.branch_lens = np.cumsum([b.shape[1] for b in outputs])
        return outputs

    def forward(self, x):
        outputs = self._forward(x)
        return torch.cat(outputs, 1)

    def relprop(self, R, alpha=2/3):
        # undo concatenation to extract branches
        bl = [0] + self.branch_lens.tolist()
        branch3x3, branch3x3dbl, branch_pool = [R[:, bl[i]:bl[i+1]] for i in range(len(bl)-1)]
        # relevance of the pooling branch
        branch_pool = self.max_pool.relprop(branch_pool, alpha)
        # relevance of the 3x3dbl branch
        branch3x3dbl = self.branch3x3dbl_3.relprop(branch3x3dbl, alpha)
        branch3x3dbl = self.branch3x3dbl_2.relprop(branch3x3dbl, alpha)
        branch3x3dbl = self.branch3x3dbl_1.relprop(branch3x3dbl, alpha)
        # relevance of the 3x3 branch
        branch3x3 = self.branch3x3.relprop(branch3x3, alpha)
        # merge branch relevances by adding them up
        return branch_pool + branch3x3dbl + branch3x3


class InceptionC(nn.Module):

    def __init__(self, in_channels, channels_7x7, conv_block=None):
        super(InceptionC, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.branch1x1 = conv_block(in_channels, 192, kernel_size=1)

        c7 = channels_7x7
        self.branch7x7_1 = conv_block(in_channels, c7, kernel_size=1)
        self.branch7x7_2 = conv_block(c7, c7, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7_3 = conv_block(c7, 192, kernel_size=(7, 1), padding=(3, 0))

        self.branch7x7dbl_1 = conv_block(in_channels, c7, kernel_size=1)
        self.branch7x7dbl_2 = conv_block(c7, c7, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7dbl_3 = conv_block(c7, c7, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7dbl_4 = conv_block(c7, c7, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7dbl_5 = conv_block(c7, 192, kernel_size=(1, 7), padding=(0, 3))

        self.branch_pool = conv_block(in_channels, 192, kernel_size=1)

        self.avg_pool = AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.branch_lens = np.zeros(4)

    def _forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch7x7 = self.branch7x7_1(x)
        branch7x7 = self.branch7x7_2(branch7x7)
        branch7x7 = self.branch7x7_3(branch7x7)

        branch7x7dbl = self.branch7x7dbl_1(x)
        branch7x7dbl = self.branch7x7dbl_2(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_3(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_4(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_5(branch7x7dbl)

        #branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.avg_pool(x)#, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch7x7, branch7x7dbl, branch_pool]
        self.branch_lens = np.cumsum([b.shape[1] for b in outputs])
        return outputs

    def forward(self, x):
        outputs = self._forward(x)
        return torch.cat(outputs, 1)

    def relprop(self, R, alpha):
        # undo concatenation to extract branches
        bl = [0] + self.branch_lens.tolist()
        branch1x1, branch7x7, branch7x7dbl, branch_pool = [R[:, bl[i]:bl[i+1]] for i in range(len(bl)-1)]
        # relevance of the pooling branch
        branch_pool = self.branch_pool.relprop(branch_pool, alpha)
        branch_pool = self.avg_pool.relprop(branch_pool, alpha)
        # relevance of the 7x7dbl branch
        branch7x7dbl = self.branch7x7dbl_5.relprop(branch7x7dbl, alpha)
        branch7x7dbl = self.branch7x7dbl_4.relprop(branch7x7dbl, alpha)
        branch7x7dbl = self.branch7x7dbl_3.relprop(branch7x7dbl, alpha)
        branch7x7dbl = self.branch7x7dbl_2.relprop(branch7x7dbl, alpha)
        branch7x7dbl = self.branch7x7dbl_1.relprop(branch7x7dbl, alpha)
        # relevance of the 7x7 branch
        branch7x7 = self.branch7x7_3.relprop(branch7x7, alpha)
        branch7x7 = self.branch7x7_2.relprop(branch7x7, alpha)
        branch7x7 = self.branch7x7_1.relprop(branch7x7, alpha)
        # relevance of the 1x1 branch
        branch1x1 = self.branch1x1.relprop(branch1x1, alpha)
        # merge relevances by adding them up
        return branch_pool + branch7x7dbl + branch7x7 + branch1x1


class InceptionD(nn.Module):

    def __init__(self, in_channels, conv_block=None):
        super(InceptionD, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.branch3x3_1 = conv_block(in_channels, 192, kernel_size=1)
        self.branch3x3_2 = conv_block(192, 320, kernel_size=3, stride=2)

        self.branch7x7x3_1 = conv_block(in_channels, 192, kernel_size=1)
        self.branch7x7x3_2 = conv_block(192, 192, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7x3_3 = conv_block(192, 192, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7x3_4 = conv_block(192, 192, kernel_size=3, stride=2)

        self.max_pool = MaxPool2d(kernel_size=3, stride=2)
        self.branch_lens = np.zeros(3)

    def _forward(self, x):
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)

        branch7x7x3 = self.branch7x7x3_1(x)
        branch7x7x3 = self.branch7x7x3_2(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_3(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_4(branch7x7x3)

        #branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)
        branch_pool = self.max_pool(x)#, kernel_size=3, stride=2)
        outputs = [branch3x3, branch7x7x3, branch_pool]
        self.branch_lens = np.cumsum([b.shape[1] for b in outputs])
        return outputs

    def forward(self, x):
        outputs = self._forward(x)
        return torch.cat(outputs, 1)

    def relprop(self, R, alpha):
        # undo concatenation to extract branches
        bl = [0] + self.branch_lens.tolist()
        branch3x3, branch7x7x3, branch_pool = [R[:, bl[i]:bl[i+1]] for i in range(len(bl)-1)]
        # relevance of the pooling branch
        branch_pool = self.max_pool.relprop(branch_pool, alpha)
        # relevance of the 7x7x3 branch
        branch7x7x3 = self.branch7x7x3_4.relprop(branch7x7x3, alpha)
        branch7x7x3 = self.branch7x7x3_3.relprop(branch7x7x3, alpha)
        branch7x7x3 = self.branch7x7x3_2.relprop(branch7x7x3, alpha)
        branch7x7x3 = self.branch7x7x3_1.relprop(branch7x7x3, alpha)
        # relevance of the 3x3 branch
        branch3x3 = self.branch3x3_2.relprop(branch3x3, alpha)
        branch3x3 = self.branch3x3_1.relprop(branch3x3, alpha)
        # merge relevances by adding them up
        return branch_pool + branch7x7x3 + branch3x3


class InceptionE(nn.Module):

    def __init__(self, in_channels, conv_block=None):
        super(InceptionE, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.branch1x1 = conv_block(in_channels, 320, kernel_size=1)

        self.branch3x3_1 = conv_block(in_channels, 384, kernel_size=1)
        self.branch3x3_2a = conv_block(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3_2b = conv_block(384, 384, kernel_size=(3, 1), padding=(1, 0))

        self.branch3x3dbl_1 = conv_block(in_channels, 448, kernel_size=1)
        self.branch3x3dbl_2 = conv_block(448, 384, kernel_size=3, padding=1)
        self.branch3x3dbl_3a = conv_block(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3dbl_3b = conv_block(384, 384, kernel_size=(3, 1), padding=(1, 0))

        self.branch_pool = conv_block(in_channels, 192, kernel_size=1)

        self.avg_pool = AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.branch_lens = np.zeros(4)
        self.branch_3x3_lens = np.zeros(2)
        self.branch_3x3dbl_lens = np.zeros(2)

    def _forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [
            self.branch3x3_2a(branch3x3),
            self.branch3x3_2b(branch3x3),
        ]
        self.branch_3x3_lens = np.cumsum([b.shape[1] for b in branch3x3])
        branch3x3 = torch.cat(branch3x3, 1)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [
            self.branch3x3dbl_3a(branch3x3dbl),
            self.branch3x3dbl_3b(branch3x3dbl),
        ]
        self.branch_3x3dbl_lens = np.cumsum([b.shape[1] for b in branch3x3dbl])
        branch3x3dbl = torch.cat(branch3x3dbl, 1)

        #branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.avg_pool(x)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        self.branch_lens = np.cumsum([b.shape[1] for b in outputs])
        return outputs

    def forward(self, x):
        outputs = self._forward(x)
        return torch.cat(outputs, 1)

    def relprop(self, R, alpha):
        # undo concatenation to extract branches
        bl = [0] + self.branch_lens.tolist()
        branch1x1, branch3x3, branch3x3dbl, branch_pool = [R[:, bl[i]:bl[i+1]] for i in range(len(bl)-1)]

        # relevance of the pooling branch
        branch_pool = self.branch_pool.relprop(branch_pool, alpha)
        branch_pool = self.avg_pool.relprop(branch_pool, alpha)
        # relevance of the 3x3dbl branch
        # undo concatention of sub-branches
        bl = [0] + self.branch_3x3dbl_lens.tolist()
        branch3x3dbl_3a, branch3x3dbl_3b = [branch3x3dbl[:, bl[i]:bl[i+1]] for i in range(len(bl)-1)]
        # sub-branch a
        branch3x3dbl_3a = self.branch3x3dbl_3a.relprop(branch3x3dbl_3a, alpha)
        branch3x3dbl_3a = self.branch3x3dbl_2.relprop(branch3x3dbl_3a, alpha)
        branch3x3dbl_3a = self.branch3x3dbl_1.relprop(branch3x3dbl_3a, alpha)
        # sub-branch b
        branch3x3dbl_3b = self.branch3x3dbl_3b.relprop(branch3x3dbl_3b, alpha)
        branch3x3dbl_3b = self.branch3x3dbl_2.relprop(branch3x3dbl_3b, alpha)
        branch3x3dbl_3b = self.branch3x3dbl_1.relprop(branch3x3dbl_3b, alpha)
        # merge sub-branches
        branch3x3dbl = branch3x3dbl_3a + branch3x3dbl_3b
        # relevance of the 3x3 branch
        # undo concatention of sub-branches
        bl = [0] + self.branch_3x3_lens.tolist()
        branch3x3_2a, branch3x3_2b = [branch3x3[:, bl[i]:bl[i+1]] for i in range(len(bl)-1)]
        # sub-branch a
        branch3x3_2a = self.branch3x3_2a.relprop(branch3x3_2a, alpha)
        branch3x3_2a = self.branch3x3_1.relprop(branch3x3_2a, alpha)
        # sub-branch b
        branch3x3_2b = self.branch3x3_2b.relprop(branch3x3_2b, alpha)
        branch3x3_2b = self.branch3x3_1.relprop(branch3x3_2b, alpha)
        # merging sub-branches
        branch3x3 = branch3x3_2a + branch3x3_2b
        # relevance of the 1x1 branch
        branch1x1 = self.branch1x1.relprop(branch1x1, alpha)
        # merge relevances by adding them up
        return branch_pool + branch3x3dbl + branch3x3 + branch1x1


class InceptionAux(nn.Module):

    def __init__(self, in_channels, num_classes, conv_block=None):
        super(InceptionAux, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.conv0 = conv_block(in_channels, 128, kernel_size=1)
        self.conv1 = conv_block(128, 768, kernel_size=5)
        self.conv1.stddev = 0.01
        self.fc = Linear2(768, num_classes)
        self.fc.stddev = 0.001

        self.adaptive_avg_pool = AdaptiveAvgPool2d((1, 1))
        self.flatten = Flatten(start_dim=1)
        self.avg_pool = AvgPool2d(kernel_size=5, stride=3)

    def forward(self, x):
        # N x 768 x 17 x 17
        x = self.avg_pool(x)#, kernel_size=5, stride=3)
        # N x 768 x 5 x 5
        x = self.conv0(x)
        # N x 128 x 5 x 5
        x = self.conv1(x)
        # N x 768 x 1 x 1
        # Adaptive average pooling
        x = self.adaptive_avg_pool(x)#, (1, 1))
        # N x 768 x 1 x 1
        x = self.flatten(x)
        # N x 768
        x = self.fc(x)
        # N x 1000
        return x

    def relprop(self, R, alpha):
        R = self.fc.relprop(R, alpha, use_epsilon=False)
        R = self.flatten.relprop(R, alpha)
        R = self.adaptive_avg_pool.relprop(R, alpha)
        R = self.conv1.relprop(R, alpha)
        R = self.conv0.relprop(R, alpha)
        R = self.avg_pool(R, alpha)
        return R


class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = BatchNorm2d(out_channels, eps=0.001)
        self.relu = ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return self.relu(x)#, inplace=True)

    def relprop(self, R, alpha=2/3):
        R = self.relu.relprop(R, alpha)
        R = self.bn.relprop(R, alpha)
        R = self.conv.relprop(R, alpha)
        return R


class InceptionV3(nn.Module):
    """Inception-V3 with no AuxLogits
    FIXME two class defs are redundant, but less screwing around with torchsript fussyness and inconsistent returns
    """

    def __init__(self, num_classes=1000, in_chans=3, drop_rate=0., global_pool='avg', aux_logits=False):
        super(InceptionV3, self).__init__()
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        self.aux_logits = aux_logits

        self.Conv2d_1a_3x3 = BasicConv2d(in_chans, 32, kernel_size=3, stride=2)
        self.Conv2d_2a_3x3 = BasicConv2d(32, 32, kernel_size=3)
        self.Conv2d_2b_3x3 = BasicConv2d(32, 64, kernel_size=3, padding=1)
        self.Pool1 = MaxPool2d(kernel_size=3, stride=2)
        self.Conv2d_3b_1x1 = BasicConv2d(64, 80, kernel_size=1)
        self.Conv2d_4a_3x3 = BasicConv2d(80, 192, kernel_size=3)
        self.Pool2 = MaxPool2d(kernel_size=3, stride=2)
        self.Mixed_5b = InceptionA(192, pool_features=32)
        self.Mixed_5c = InceptionA(256, pool_features=64)
        self.Mixed_5d = InceptionA(288, pool_features=64)
        self.Mixed_6a = InceptionB(288)
        self.Mixed_6b = InceptionC(768, channels_7x7=128)
        self.Mixed_6c = InceptionC(768, channels_7x7=160)
        self.Mixed_6d = InceptionC(768, channels_7x7=160)
        self.Mixed_6e = InceptionC(768, channels_7x7=192)
        if aux_logits:
            self.AuxLogits = InceptionAux(768, num_classes)
        else:
            self.AuxLogits = None
        self.Mixed_7a = InceptionD(768)
        self.Mixed_7b = InceptionE(1280)
        self.Mixed_7c = InceptionE(2048)
        self.feature_info = [
            dict(num_chs=64, reduction=2, module='Conv2d_2b_3x3'),
            dict(num_chs=192, reduction=4, module='Conv2d_4a_3x3'),
            dict(num_chs=288, reduction=8, module='Mixed_5d'),
            dict(num_chs=768, reduction=16, module='Mixed_6e'),
            dict(num_chs=2048, reduction=32, module='Mixed_7c'),
        ]

        self.num_features = 2048
        self.global_pool, self.fc = create_classifier(self.num_features, self.num_classes, pool_type=global_pool)

        for m in self.modules():
            if isinstance(m, Conv2d) or isinstance(m, Linear):
                stddev = m.stddev if hasattr(m, 'stddev') else 0.1
                trunc_normal_(m.weight, std=stddev)
            elif isinstance(m, BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        module_map = {k: i for i, (k, _) in enumerate(flatten_modules(self.named_children(), prefix=()))}
        module_map.pop(('fc',))

        def _matcher(name):
            if any([name.startswith(n) for n in ('Conv2d_1', 'Conv2d_2')]):
                return 0
            elif any([name.startswith(n) for n in ('Conv2d_3', 'Conv2d_4')]):
                return 1
            else:
                for k in module_map.keys():
                    if k == tuple(name.split('.')[:len(k)]):
                        return module_map[k]
                return float('inf')
        return _matcher

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        assert not enable, 'gradient checkpointing not supported'

    @torch.jit.ignore
    def get_classifier(self):
        return self.fc

    def reset_classifier(self, num_classes, global_pool='avg'):
        self.num_classes = num_classes
        self.global_pool, self.fc = create_classifier(self.num_features, self.num_classes, pool_type=global_pool)

    def forward_preaux(self, x):
        x = self.Conv2d_1a_3x3(x)  # N x 32 x 149 x 149
        x = self.Conv2d_2a_3x3(x)  # N x 32 x 147 x 147
        x = self.Conv2d_2b_3x3(x)  # N x 64 x 147 x 147
        x = self.Pool1(x)  # N x 64 x 73 x 73
        x = self.Conv2d_3b_1x1(x)  # N x 80 x 73 x 73
        x = self.Conv2d_4a_3x3(x)  # N x 192 x 71 x 71
        x = self.Pool2(x)  # N x 192 x 35 x 35
        x = self.Mixed_5b(x)  # N x 256 x 35 x 35
        x = self.Mixed_5c(x)  # N x 288 x 35 x 35
        x = self.Mixed_5d(x)  # N x 288 x 35 x 35
        x = self.Mixed_6a(x)  # N x 768 x 17 x 17
        x = self.Mixed_6b(x)  # N x 768 x 17 x 17
        x = self.Mixed_6c(x)  # N x 768 x 17 x 17
        x = self.Mixed_6d(x)  # N x 768 x 17 x 17
        x = self.Mixed_6e(x)  # N x 768 x 17 x 17
        return x

    def forward_postaux(self, x):
        x = self.Mixed_7a(x)  # N x 1280 x 8 x 8
        x = self.Mixed_7b(x)  # N x 2048 x 8 x 8
        x = self.Mixed_7c(x)  # N x 2048 x 8 x 8
        return x

    def forward_features(self, x):
        x = self.forward_preaux(x)
        x = self.forward_postaux(x)
        return x

    def forward_head(self, x):
        x = self.global_pool(x)
        if self.drop_rate > 0:
            x = F.dropout(x, p=self.drop_rate, training=self.training)
        x = self.fc(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x

    def relprop(self, R, alpha=2/3):
        R = self.fc.relprop(R, alpha, use_epsilon=True)
        R = self.global_pool.relprop(R, alpha)
        R = self.Mixed_7c.relprop(R, alpha)
        R = self.Mixed_7b.relprop(R, alpha)
        R = self.Mixed_7a.relprop(R, alpha)

        R = self.Mixed_6e.relprop(R, alpha)
        R = self.Mixed_6d.relprop(R, alpha)
        R = self.Mixed_6c.relprop(R, alpha)
        R = self.Mixed_6b.relprop(R, alpha)
        R = self.Mixed_6a.relprop(R, alpha)

        R = self.Mixed_5d.relprop(R, alpha)
        R = self.Mixed_5c.relprop(R, alpha)
        R = self.Mixed_5b.relprop(R, alpha)

        R = self.Pool2.relprop(R, alpha)
        R = self.Conv2d_4a_3x3.relprop(R, alpha)
        R = self.Conv2d_3b_1x1.relprop(R, alpha)

        R = self.Pool1.relprop(R, alpha)
        R = self.Conv2d_2b_3x3.relprop(R, alpha)
        R = self.Conv2d_2a_3x3.relprop(R, alpha)
        R = self.Conv2d_1a_3x3.relprop(R, alpha)

        return R


class InceptionV3Aux(InceptionV3):
    """InceptionV3 with AuxLogits
    """

    def __init__(self, num_classes=1000, in_chans=3, drop_rate=0., global_pool='avg', aux_logits=True):
        super(InceptionV3Aux, self).__init__(
            num_classes, in_chans, drop_rate, global_pool, aux_logits)

    def forward_features(self, x):
        x = self.forward_preaux(x)
        aux = self.AuxLogits(x) if self.training else None
        x = self.forward_postaux(x)
        return x, aux

    def forward(self, x):
        x, aux = self.forward_features(x)
        x = self.forward_head(x)
        return x, aux


def _create_inception_v3(variant, pretrained=False, **kwargs):
    pretrained_cfg = resolve_pretrained_cfg(variant, pretrained_cfg=kwargs.pop('pretrained_cfg', None))
    aux_logits = kwargs.pop('aux_logits', False)
    if aux_logits:
        assert not kwargs.pop('features_only', False)
        model_cls = InceptionV3Aux
        load_strict = pretrained_cfg['has_aux']
    else:
        model_cls = InceptionV3
        load_strict = not pretrained_cfg['has_aux']

    return build_model_with_cfg(
        model_cls, variant, pretrained,
        pretrained_cfg=pretrained_cfg,
        pretrained_strict=load_strict,
        **kwargs)


@register_model
def inception_v3(pretrained=False, **kwargs):
    # original PyTorch weights, ported from Tensorflow but modified
    model = _create_inception_v3('inception_v3', pretrained=pretrained, **kwargs)
    return model
