"""
BotNet50 based on: https://github.com/BIGBALLON/distribuuuu/blob/master/distribuuuu/models/botnet.py
"""
import utils
#from torchvision.models.inception import inception_v3, InceptionOutputs
import torch
#from torch import einsum
from torch import nn
from copy import deepcopy
from pytorch_LRP.resnet import resnet50
from pytorch_LRP.layers import *
from pytorch_LRP.transformer.transformer_layers import *
from pytorch_LRP.inception import inception_v3, InceptionOutputs
from pytorch_LRP.transformer import vit
from einops import rearrange
from torchvision import transforms
import numpy as np


class expand_dim(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t, k):
        """
        Expand dims for t at dim to k
        """
        t = t.unsqueeze(dim=self.dim)
        expand_shape = [-1] * len(t.shape)
        expand_shape[self.dim] = k
        return t.expand(*expand_shape)

    def relprop(self, R, **kwargs):
        return R.sum(dim=self.dim)


class rel_to_abs(nn.Module):

    def forward(self, x):
        """
        x: [B, Nh * H, L, 2L - 1]
        Convert relative position between the key and query to their absolute position respectively.
        Tensowflow source code in the appendix of: https://arxiv.org/pdf/1904.09925.pdf
        """
        B, Nh, L, _ = x.shape
        # pad to shift from relative to absolute indexing
        col_pad = torch.zeros((B, Nh, L, 1)).cuda()
        self.len_x = x.shape[3]
        x = torch.cat((x, col_pad), dim=3)
        self.x_shape = x.shape
        flat_x = torch.reshape(x, (B, Nh, L * 2 * L))
        flat_pad = torch.zeros((B, Nh, L - 1)).cuda()
        self.len_flat_x = flat_x.shape[2]
        flat_x = torch.cat((flat_x, flat_pad), dim=2)
        # Reshape and slice out the padded elements
        self.flat_x_shape = flat_x.shape
        final_x = torch.reshape(flat_x, (B, Nh, L + 1, 2 * L - 1))
        self.pad_d2 = 1
        self.pad_d3 = L
        return final_x[:, :, :L, L - 1 :]

    def relprop(self, R, **kwargs):
        R_padded = torch.zeros(list(R.shape[:2]) + [R.shape[2]+self.pad_d2, R.shape[3]+self.pad_d3]).to(R.device)
        R_padded[:,:,:R.shape[2],:R.shape[3]] = R
        R_padded = R_padded.reshape(self.flat_x_shape)
        R_flat_x = R_padded[:,:,:self.len_flat_x]
        R_x = R_flat_x.reshape(self.x_shape)
        return R_x[:,:,:,:self.len_x]


class relative_logits_1d(nn.Module):

    def __init__(self):
        super().__init__()
        self.matmul = einsum("b n h w d, m d -> b n h w m")
        self.expand_dim = expand_dim(dim=3)
        self.rel_to_abs = rel_to_abs()

    def forward(self, q, rel_k):
        """
        q: [B, Nh, H, W, d]
        rel_k: [2W - 1, d]
        Computes relative logits along one dimension.
        The details of relative position is explained in: https://arxiv.org/pdf/1803.02155.pdf
        """
        B, Nh, H, W, _ = q.shape
        rel_logits = self.matmul([q, rel_k])
        # Collapse height and heads
        self.rel_logits_shape1 = rel_logits.shape
        rel_logits = torch.reshape(rel_logits, (-1, Nh * H, W, 2 * W - 1))
        rel_logits = self.rel_to_abs(rel_logits)
        self.rel_logits_shape2 = rel_logits.shape
        rel_logits = torch.reshape(rel_logits, (-1, Nh, H, W, W))
        rel_logits = self.expand_dim(rel_logits, k=H)
        return rel_logits

    def relprop(self, R, **kwargs):
        R = self.expand_dim.relprop(R, **kwargs)
        R = R.reshape(self.rel_logits_shape2)
        R = self.rel_to_abs.relprop(R, **kwargs)
        R = R.reshape(self.rel_logits_shape1)
        R = self.matmul.relprop(R, **kwargs)
        return R



class AbsPosEmb(nn.Module):
    def __init__(self, height, width, dim_head):
        super().__init__()
        # assert height == width
        scale = dim_head ** -0.5
        self.height = nn.Parameter(torch.randn(height, dim_head) * scale)
        self.width = nn.Parameter(torch.randn(width, dim_head) * scale)
        self.matmul = einsum("b h i d, j d -> b h i j")

    def forward(self, q):
        emb = rearrange(self.height, "h d -> h () d") + rearrange(
            self.width, "w d -> () w d"
        )
        emb = rearrange(emb, " h w d -> (h w) d")
        logits = self.matmul([q, emb])
        return logits

    def relprop(self, R, alpha=None):
        (R_q, R_emb) = self.matmul.relprop(R, alpha=alpha)
        return R_q


class RelPosEmb(nn.Module):
    def __init__(self, height, width, dim_head):
        super().__init__()
        # assert height == width
        scale = dim_head ** -0.5
        self.height = height
        self.width = width
        self.rel_height = nn.Parameter(torch.randn(height * 2 - 1, dim_head) * scale)
        self.rel_width = nn.Parameter(torch.randn(width * 2 - 1, dim_head) * scale)
        self.add = Add()
        self.relative_logits_1d_1 = relative_logits_1d()
        self.relative_logits_1d_2 = relative_logits_1d()

    def forward(self, q):
        h = self.height
        w = self.width

        q = rearrange(q, "b h (x y) d -> b h x y d", x=h, y=w)
        rel_logits_w = self.relative_logits_1d_1(q, self.rel_width)
        rel_logits_w = rearrange(rel_logits_w, "b h x i y j-> b h (x y) (i j)")

        q = rearrange(q, "b h x y d -> b h y x d")
        rel_logits_h = self.relative_logits_1d_2(q, self.rel_height)
        self.last_i, self.last_j = rel_logits_h.shape[3], rel_logits_h.shape[-1]
        rel_logits_h = rearrange(rel_logits_h, "b h x i y j -> b h (y x) (j i)")
        return self.add([rel_logits_w, rel_logits_h])

    def relprop(self, R, **kwargs):
        R_logits_w, R_logits_h = self.add.relprop(R, **kwargs)
        R_logits_h = rearrange(R_logits_h, "b h (y x) (j i) -> b h x i y j", x=self.height, y=self.width, j=self.last_j, i=self.last_i)
        R_logits_h = self.relative_logits_1d_2.relprop(R_logits_h, **kwargs)
        R_q = rearrange(R_logits_h, "b h y x d -> b h x y d")
        R_logits_w = rearrange("b h (x y) (i j) -> b h x i y j", x=self.height, y=self.width, j=self.last_j, i=self.last_i)
        R_q += self.relative_logits_1d_1(R_logits_w, **kwargs)
        R_q = rearrange("b h x y d -> b h (x y) d")
        return R_q


class BoTBlock(nn.Module):
    def __init__(
        self,
        dim,
        fmap_size,
        dim_out,
        stride=1,
        heads=4,
        proj_factor=4,
        dim_qk=128,
        dim_v=128,
        rel_pos_emb=False,
        activation=ReLU(),
    ):
        """
        dim: channels in feature map
        dim_out: output channels for feature map
        """
        super().__init__()
        if dim != dim_out or stride != 1:
            self.shortcut = Sequential(
                Conv2d(dim, dim_out, kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(dim_out),
                activation,
            )
        else:
            self.shortcut = Identity()

        bottleneck_dimension = dim_out // proj_factor  # from 2048 to 512
        attn_dim_out = heads * dim_v

        self.net = Sequential(
            Conv2d(dim, bottleneck_dimension, kernel_size=1, stride=1, bias=False),
            BatchNorm2d(bottleneck_dimension),
            activation,
            MHSA(
                dim=bottleneck_dimension,
                fmap_size=fmap_size,
                heads=heads,
                dim_qk=dim_qk,
                dim_v=dim_v,
                rel_pos_emb=rel_pos_emb,
            ),
            AvgPool2d((2, 2)) if stride == 2 else Identity(),  # same padding
            BatchNorm2d(attn_dim_out),
            activation,
            Conv2d(attn_dim_out, dim_out, kernel_size=1, stride=1, bias=False),
            BatchNorm2d(dim_out),
        )

        nn.init.zeros_(
            self.net[-1].weight
        )  # last batch norm uses zero gamma initializer
        self.activation = activation

    def forward(self, featuremap):
        shortcut = self.shortcut(featuremap)
        featuremap = self.net(featuremap)
        featuremap += shortcut
        return self.activation(featuremap)

    def relprop(self, R, alpha=None):
        R = self.activation.relprop(R, alpha=alpha)
        R = self.net.relprop(R, alpha=alpha) + self.shortcut.relprop(R, alpha=alpha)
        return R


class MHSA(nn.Module):
    def __init__(
        self, dim, fmap_size, heads=4, dim_qk=128, dim_v=128, rel_pos_emb=False
    ):
        """
        dim: number of channels of feature map
        fmap_size: [H, W]
        dim_qk: vector dimension for q, k
        dim_v: vector dimension for v (not necessarily the same with q, k)
        """
        super().__init__()
        self.scale = dim_qk ** -0.5
        self.heads = heads
        out_channels_qk = heads * dim_qk
        out_channels_v = heads * dim_v

        self.to_qk = Conv2d(
            dim, out_channels_qk * 2, 1, bias=False
        )  # 1*1 conv to compute q, k
        self.to_v = Conv2d(
            dim, out_channels_v, 1, bias=False
        )  # 1*1 conv to compute v
        self.softmax = Softmax(dim=-1)

        height, width = fmap_size
        if rel_pos_emb:
            self.pos_emb = RelPosEmb(height, width, dim_qk)
        else:
            self.pos_emb = AbsPosEmb(height, width, dim_qk)

        self.matmul1 = einsum("b h x d, b h y d -> b h x y")
        self.matmul2 = einsum("b h x y, b h y d -> b h x d")

        self.add = Add()

    def forward(self, featuremap):
        """
        featuremap: [B, d_in, H, W]
        Output: [B, H, W, head * d_v]
        """
        heads = self.heads
        # 
        B, C, H, W = featuremap.shape
        self.last_H = H
        q, k = self.to_qk(featuremap).chunk(2, dim=1)
        v = self.to_v(featuremap)
        q, k, v = map(
            lambda x: rearrange(x, "B (h d) H W -> B h (H W) d", h=heads), (q, k, v)
        )

        q = q*self.scale

        logits = self.matmul1([q, k])
        # TODO
        logits = self.add([logits, self.pos_emb(q)])

        weights = self.softmax(logits)
        attn_out = self.matmul2([weights, v])
        attn_out = rearrange(attn_out, "B h (H W) d -> B (h d) H W", H=H)

        return attn_out

    def relprop(self, R, alpha=None):
        R = rearrange(R, "B (h d) H W -> B h (H W) d", H=self.last_H, h=self.heads)
        (R_weights, R_v) = self.matmul2.relprop(R, alpha=alpha)

        R_logits = self.softmax.relprop(R_weights, alpha=alpha)
        R_logits, R_q = self.add.relprop(R_logits, alpha=alpha)
        R_q = self.pos_emb.relprop(R_q)

        (R_q_mul, R_k) = self.matmul1.relprop(R_logits, alpha=alpha)
        R_q += R_q_mul
        # TODO: scaling in relptop???
        R_q /= self.scale

        R_q, R_k, R_v = map(
            lambda x: rearrange(x, "B h (H W) d -> B (h d) H W", H=self.last_H, h=self.heads), (R_q, R_k, R_v)
        )
        R_v = self.to_v.relprop(R_v, alpha=alpha)
        R_featuremap = self.to_qk.relprop(torch.cat([R_q, R_k], dim=1), alpha=alpha)
        return R_featuremap + R_v


class BoTStack(nn.Module):
    def __init__(
        self,
        dim,
        fmap_size,
        dim_out=2048,
        heads=4,
        proj_factor=4,
        num_layers=3,
        stride=2,
        dim_qk=128,
        dim_v=128,
        rel_pos_emb=False,
        activation=ReLU(),
    ):
        """
        dim: channels in feature map
        fmap_size: [H, W]
        """
        super().__init__()

        self.dim = dim
        self.fmap_size = fmap_size

        layers = []

        for i in range(num_layers):
            is_first = i == 0
            dim = dim if is_first else dim_out

            fmap_divisor = 2 if stride == 2 and not is_first else 1
            layer_fmap_size = tuple(map(lambda t: t // fmap_divisor, fmap_size))

            layers.append(
                BoTBlock(
                    dim=dim,
                    fmap_size=layer_fmap_size,
                    dim_out=dim_out,
                    stride=stride if is_first else 1,
                    heads=heads,
                    proj_factor=proj_factor,
                    dim_qk=dim_qk,
                    dim_v=dim_v,
                    rel_pos_emb=rel_pos_emb,
                    activation=activation,
                )
            )

        self.net = Sequential(*layers)

    def forward(self, x):
        _, c, h, w = x.shape
        assert c == self.dim, f"assert {c} == self.dim {self.dim}"
        assert h == self.fmap_size[0] and w == self.fmap_size[1]
        return self.net(x)

    def relprop(self, R, alpha=None):
        R = self.net.relprop(R, alpha=alpha)
        return R


class BotNet50(nn.Module):

    def __init__(self, n_classes=None, size_in=299, final_act=False):
        super().__init__()
        resnet = resnet50()
        self.conv = Sequential(*list(resnet.children())[:-3])
        self.mhsa = BoTStack(
            dim=1024, 
            fmap_size=(round(0.5+size_in/(2**4)), round(0.5+size_in/(2**4))), 
            stride=1, 
            rel_pos_emb=False#True
        )
        self.final_pool = Sequential(AdaptiveAvgPool2d((1, 1)), Flatten(1))
        self.f_out = 2048
        if n_classes is not None:
            self.head = Sequential(Linear(2048, n_classes), LogSoftmax(dim=1) if final_act else Identity())
        else:
            self.head = None

    def forward(self, x):
        y = self.conv(x)
        y = self.mhsa(y)
        y = self.final_pool(y)
        if self.head is not None:
            y = self.head(y)
        return y

    def relprop(self, R, alpha=None):
        if self.head is not None:
            R = self.head.relprop(R, alpha=alpha)
        R = self.final_pool.relprop(R, alpha=alpha)
        R = self.mhsa.relprop(R, alpha=alpha)
        R = self.conv.relprop(R, alpha=alpha)
        return R


class Inception(nn.Module):

    def __init__(self, n_classes=None):
        super().__init__()
        if n_classes is not None:
            self.model = inception_v3(num_classes=n_classes, init_weights=True)
        else:
            self.model = inception_v3(num_classes=1, init_weights=True)
            # dropout will be performed in GasHis head
            self.model.dropout = Identity()
            self.model.fc = Identity()
        self.f_out = 2048

    def relprop(self, R, alpha=2/3):
        return self.model.relprop(R, alpha)

    def forward(self, x):
        y = self.model(x)
        if isinstance(y, InceptionOutputs):
            return y[0]
        return y


class GasHisTransformer(nn.Module):

    def __init__(self, hparams, gim=None, lim=None, final_act=False, dropout_gim=True, dropout_lim=True):
        super().__init__()
        if gim is None:
            gim = BotNet50(size_in=299)
        if lim is None:
            lim = Inception()
        self.hparams = hparams
        self.f_GIM = gim
        self.f_LIM = lim
        feature_len = self.f_GIM.f_out + self.f_LIM.f_out
        self.dropout = Dropout(p=hparams.dropout)
        self.dropout_gim = dropout_gim
        self.dropout_lim = dropout_lim
        self.lin_out = Linear(feature_len, hparams.num_classes)
        self.classifier = Sequential(
            self.lin_out, LogSoftmax(dim=1) if final_act else Identity()
        )

    def forward(self, x):
        features_GIM = self.f_GIM(x)
        features_LIM = self.f_LIM(x)
        if self.dropout_gim:
            features_GIM = self.dropout(features_GIM)
        if self.dropout_lim:
            features_LIM = self.dropout(features_LIM)
        self.len_F_GIM = features_GIM.shape[1]
        features = torch.cat([features_GIM, features_LIM], dim=1)
        return self.classifier(features)

    def relprop(self, R, alpha=None, uniform_size=True):
        R = self.classifier.relprop(R, alpha)
        R_GIM = R[:,:self.len_F_GIM]
        R_LIM = R[:,self.len_F_GIM:]
        if self.dropout_lim:
            R_LIM = self.dropout.relprop(R_LIM, alpha=alpha)
        if self.dropout_gim:
            R_GIM = self.dropout.relprop(R_GIM, alpha=alpha)
        R_GIM = self.f_GIM.relprop(R_GIM, alpha=alpha) 
        R_LIM = self.f_LIM.relprop(R_LIM, alpha=alpha)
        if R_GIM.shape != R_LIM.shape:
            # if they differ in input size, we resize the GIM input to that of the LIM input
            if uniform_size:
                relevance_mass_before = R_GIM.sum()
                R_GIM = transforms.functional.resize(R_GIM, R_LIM.shape)
                # assuring, that the sum of relevance remains
                if R_GIM.sum() != 0 and relevance_refactoring != 0:
                    relevance_refactoring = relevance_mass_before / R_GIM.sum()
                    R_GIM *= relevance_refactoring
            # alternatively one could receive the different branches of inputs
            else:
                return R_GIM, R_LIM
        return R_GIM + R_LIM

    def relevance_distribution(self, x, y=None):
        # forward pass
        self.eval()
        self.zero_grad()
        logit = self(x)
        # one hot encoding
        if y is None:
            pred = torch.argmax(logit.data, dim=1)
        else:
            # asking for a sequence of classes, e.g. GT-annotations
            if isinstance(y, np.ndarray):
                y = torch.from_numpy(y_class)
            if isinstance(y, torch.Tensor):
                if len(y.shape) == 1:
                    if y.shape[0] == x_in.shape[0]:
                        y = y[:, None]
                    else:
                        raise ValueError('Either submit a class for each input or one class for all.')
                elif y.shape != (len(x_in), 1):
                        raise ValueError('Either submit a class for each input or one class for all.')
            # asking for a single class applying to all input
            else:
                y= torch.ones((len(x_in), 1)) * y_class
            pred = y.cpu()
        T = pred == torch.arange(self.hparams.num_classes).type(torch.FloatTensor)
        T = torch.autograd.Variable(T).to(self.get_device())
        # relevance distribution in ensemble
        R = self.lin_out.relprop(T, None, use_epsilon=True) #Note: using epsilon rule for completeness property
        R_GIM = R[:,:self.f_GIM.f_out]
        R_LIM = R[:,self.f_GIM.f_out:]
        return R_GIM.sum(), R_LIM.sum()


class Squeeze(nn.Module):

    def __init__(self, dim=None, start_dim=1):
        super().__init__()
        self.dim = dim
        self.start_dim = start_dim

    def forward(self, x):
        if self.dim is not None:
            return torch.squeeze(x, dim=self.dim)
        self.unsqueezed_shape = x.shape
        if self.start_dim is not None:
            while np.sum(np.array(x.shape[self.start_dim:]) == 1) > 0:
                for dim in range(self.start_dim, len(x.shape)):
                    if x.shape[dim] == 1:
                        break
                x = torch.squeeze(x, dim=dim)
        else:
            x = torch.squeeze(x)
        return x

    def relprop(self, R, *args, **kwargs):
        if self.dim is not None:
            return torch.unsqueeze(R, dim=self.dim)
        return torch.reshape(R, self.unsqueezed_shape)


class ResNet50InceptionV3(GasHisTransformer):

    def __init__(self, hparams):
        rn50 = resnet50()
        rn50 = Sequential(*list(rn50.children())[:-1], Squeeze())
        rn50.f_out = 2048
        hparams.dropout = 0.5
        super().__init__(hparams=hparams,
                         gim=Inception(),
                         lim=rn50)


class Resize(nn.Module):

    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        self.prev_shape = x.shape[2:]
        return transforms.functional.resize(x, self.shape)

    def relprop(self, R, *args, **kwargs):
        return transforms.functional.resize(R, self.prev_shape)


class ViTInceptionV3(GasHisTransformer):

    def __init__(self, hparams):
        ViT = vit.VisionTransformer(
            img_size=hparams.input_size, patch_size=hparams.patch_size, in_chans=3, num_classes=hparams.num_classes,
            embed_dim=hparams.dim_model, depth=hparams.n_layers, num_heads=hparams.n_heads,
            mlp_ratio=hparams.mlp_ratio, drop_rate=hparams.dropout, attn_drop_rate=hparams.dropout_attention
        )
        f_out = ViT.head.in_features#hparams.dim_model
        ViT.head = Identity()
        ViT = Sequential(Resize([hparams.input_size, hparams.input_size]), ViT)
        ViT.f_out = f_out
        # TODO
        #f_out = vit.head.output_dim
        lim = Sequential(Resize([299, 299]), Inception())
        lim.f_out = 2048
        hparams.dropout = 0.5
        super().__init__(hparams=hparams,
                         lim=lim,
                         gim=ViT, dropout_gim=False)


class ViTCInceptionV3(GasHisTransformer):

    def __init__(self, hparams, **kwargs):
        ViT_C = vit.VisionTransformerConvStem(hparams, **kwargs)
        ViT_C.transformer.head = Identity()
        ViT_C = Sequential(Resize([hparams.input_size, hparams.input_size]), ViT_C)
        ViT_C.f_out = hparams.dim_model
        lim = Sequential(Resize([299, 299]), Inception())
        lim.f_out = 2048
        hparams.dropout = 0.5
        super().__init__(hparams=hparams,
                         lim=lim,
                         gim=ViT_C, dropout_gim=False)

