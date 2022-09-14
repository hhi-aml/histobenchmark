import timm.models as timm_models
import torch
from torch import nn
from torchvision.transforms import Resize


class Hyrbid(nn.Module):

    def __init__(self, hparams, m1, m2):
        super().__init__()
        self.m1 = m1
        self.m2 = m2
        self.dim_features = sum([m.hybrid_dim for m in (self.m1, self.m2)])
        self.classifier = nn.Linear(self.dim_features, hparams.num_classes)

    def forward(self, x):
        features = torch.cat([m(x) for m in (self.m1, self.m2)], dim=1)
        return self.classifier(features)


class ConvNeXtT_InceptionV3(Hyrbid):

    def __init__(self, hparams):
        convnextT = timm_models.convnext_tiny_in22k(pretrained=hparams.use_pt_weights)
        convnextT.head.fc = nn.Identity()
        convnextT.hybrid_dim = 768
        iv3 = timm_models.inception_v3(pretrained=hparams.use_pt_weights)
        iv3.hybrid_dim = 2048
        iv3.fc = nn.Identity()
        super().__init__(hparams, convnextT, iv3)
        self.resize1 = Resize([224,224])
        self.resize2 = Resize([299,299])

    def forward(self, x):
        f1 = self.m1(self.resize1(x))
        f2 = self.m2(self.resize2(x))
        features = torch.cat([f1, f2], dim=1)
        return self.classifier(features)


class ViTT_ConvNeXtT(Hyrbid):

    def __init__(self, hparams):
        vitT = timm_models.vit_tiny_patch16_224_in21k(pretrained=hparams.use_pt_weights)
        vitT.head = nn.Identity()
        vitT.hybrid_dim = 192
        convnextT = timm_models.convnext_tiny_in22k(pretrained=hparams.use_pt_weights)
        convnextT.head.fc = nn.Identity()
        convnextT.hybrid_dim = 768
        super().__init__(hparams, vitT, convnextT)


class ViTL_ConvNeXtL(Hyrbid):

    def __init__(self, hparams):
        vitL = timm_models.vit_large_patch16_224_in21k(pretrained=hparams.use_pt_weights)
        vitL.head = nn.Identity()
        vitL.hybrid_dim = 1024
        convnextL = timm_models.convnext_large_in22k(pretrained=hparams.use_pt_weights)
        convnextL.head.fc = nn.Identity()
        convnextL.hybrid_dim = 1536
        super().__init__(hparams, vitL, convnextL)


class SwinT_ConvNeXtT(Hyrbid):

    def __init__(self, hparams):
        swinT = timm_models.swin_tiny_patch4_window7_224(pretrained=hparams.use_pt_weights)
        swinT.head = nn.Identity()
        swinT.hybrid_dim = 768
        convnextT = timm_models.convnext_tiny_in22k(pretrained=hparams.use_pt_weights)
        convnextT.head.fc = nn.Identity()
        convnextT.hybrid_dim = 768
        super().__init__(hparams, swinT, convnextT)


class SwinL_ConvNeXtL(Hyrbid):

    def __init__(self, hparams):
        swinL = timm_models.swin_large_patch4_window7_224_in22k(pretrained=hparams.use_pt_weights)
        swinL.head = nn.Identity()
        swinL.hybrid_dim = 1536
        convnextL = timm_models.convnext_large_in22k(pretrained=hparams.use_pt_weights)
        convnextL.head.fc = nn.Identity()
        convnextL.hybrid_dim = 1536
        super().__init__(hparams, swinL, convnextL)
