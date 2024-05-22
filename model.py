
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import timm
import torch

class Encoder(nn.Module):

    def __init__(self, out_dim=512,feat_size=56,padding=False):
        super(Encoder, self).__init__()
        self.out_dim = out_dim
        self.st_size = feat_size
        self.ae_size = feat_size
        self.padding = 1 if padding else 0
        self.conv1 = nn.Conv2d(3, 256, kernel_size=4, stride=1, padding=3*self.padding)
        self.conv2 = nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=3*self.padding)
        self.conv3 = nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0)
        self.conv4 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1*self.padding)
        self.conv5 = nn.Conv2d(512, self.out_dim, kernel_size=4, stride=1, padding=0)
        self.conv6 = nn.Conv2d(self.out_dim, self.out_dim, kernel_size=1, stride=1, padding=0)
        self.avgpool1 = nn.AvgPool2d(kernel_size=2, stride=2, padding=1*self.padding)
        self.avgpool2 = nn.AvgPool2d(kernel_size=2, stride=2, padding=1*self.padding)
        self.conv_ae = nn.Sequential(
            nn.Conv2d(self.out_dim, self.out_dim, kernel_size=1, stride=1, padding=1),
            # nn.Conv2d(self.out_dim, self.out_dim, kernel_size=4, stride=2, padding=1),
        )
        self.conv_st = nn.Sequential(
            nn.Conv2d(self.out_dim, self.out_dim, kernel_size=1, stride=1, padding=1),
            # nn.Conv2d(self.out_dim, self.out_dim, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.avgpool1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.avgpool2(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.conv5(x)
        x = F.relu(x)
        x = self.conv6(x)
        st = self.conv_st(x)
        st = F.interpolate(st, size=self.st_size, mode='bilinear', align_corners=False)
        ae = self.conv_ae(x)
        ae = F.interpolate(ae, size=self.ae_size, mode='bilinear', align_corners=False)
        return st,ae
    
def get_ae():
    return nn.Sequential(
        # encoder
        nn.Conv2d(3, 32, 4, 2, 1), nn.ReLU(inplace=True),
        nn.Conv2d(32, 32, 4, 2, 1), nn.ReLU(inplace=True),
        nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(inplace=True),
        nn.Conv2d(64, 64, 4, 2, 1), nn.ReLU(inplace=True),
        nn.Conv2d(64, 64, 4, 2, 1), nn.ReLU(inplace=True),
        nn.Conv2d(64, 64, 8),
        # decoder
        nn.Upsample(3, mode='bilinear'),
        nn.Conv2d(64, 64, 4, 1, 2), nn.ReLU(inplace=True),
        nn.Upsample(8, mode='bilinear'),
        nn.Conv2d(64, 64, 4, 1, 2), nn.ReLU(inplace=True),
        nn.Upsample(15, mode='bilinear'),
        nn.Conv2d(64, 64, 4, 1, 2), nn.ReLU(inplace=True),
        nn.Upsample(32, mode='bilinear'),
        nn.Conv2d(64, 64, 4, 1, 2), nn.ReLU(inplace=True),
        nn.Upsample(63, mode='bilinear'),
        nn.Conv2d(64, 64, 4, 1, 2), nn.ReLU(inplace=True),
        nn.Upsample(127, mode='bilinear'),
        nn.Conv2d(64, 64, 4, 1, 2), nn.ReLU(inplace=True),
        nn.Upsample(56, mode='bilinear'),
        nn.Conv2d(64, 64, 3, 1, 1), nn.ReLU(inplace=True),
        nn.Conv2d(64, 512, 3, 1, 1)
    )

class ResNetTeacher(nn.Module):
    def __init__(self,out_dim=512,feat_size=56):
        super(ResNetTeacher, self).__init__()
        self.out_dim = out_dim
        self.feat_size = feat_size

        self.proj = nn.Conv2d(1024+512, self.out_dim, kernel_size=1, stride=1, padding=0)
        self.proj.requires_grad_(False)

    def forward(self, x):
        x1,x2,x3 = x
        concat_feat = []
        for feat in [x2,x3]:
            feat = F.interpolate(feat, size=self.feat_size, mode='bilinear',align_corners=False)
            concat_feat.append(feat)
        concat_feat = torch.cat(concat_feat,dim=1)
        proj_feat = self.proj(concat_feat)
        proj_feat = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)(proj_feat)
        return proj_feat
    


class _ImagePool(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = _ConvBnReLU(in_ch, out_ch, 1, 1, 0, 1)

    def forward(self, x):
        _, _, H, W = x.shape
        h = self.pool(x)
        h = self.conv(h)
        h = F.interpolate(h, size=(H, W), mode="bilinear", align_corners=False)
        return h
    
class _ConvBnReLU(nn.Sequential):
    """
    Cascade of 2D convolution, batch norm, and ReLU.
    """

    def __init__(
            self, in_ch, out_ch, kernel_size, stride, padding, dilation, relu=True
    ):
        super(_ConvBnReLU, self).__init__()
        self.add_module(
            "conv",
            nn.Conv2d(
                in_ch, out_ch, kernel_size, stride, padding, dilation, bias=False
            ),
        )
        self.add_module("bn", nn.BatchNorm2d(out_ch, eps=1e-5, momentum=0.999))

        if relu:
            self.add_module("relu", nn.ReLU())

class _Bottleneck(nn.Module):
    """
    Bottleneck block of MSRA ResNet.
    """

    def __init__(self, in_ch, out_ch, stride, dilation, downsample):
        super(_Bottleneck, self).__init__()
        mid_ch = out_ch // 4#_BOTTLENECK_EXPANSION
        self.reduce = _ConvBnReLU(in_ch, mid_ch, 1, stride, 0, 1, True)
        self.conv3x3 = _ConvBnReLU(mid_ch, mid_ch, 3, 1, dilation, dilation, True)
        self.increase = _ConvBnReLU(mid_ch, out_ch, 1, 1, 0, 1, False)
        self.shortcut = (
            _ConvBnReLU(in_ch, out_ch, 1, stride, 0, 1, False)
            if downsample
            else lambda x: x  # identity
        )

    def forward(self, x):
        h = self.reduce(x)
        h = self.conv3x3(h)
        h = self.increase(h)
        h += self.shortcut(x)
        return F.relu(h)


class _ResLayer(nn.Sequential):
    """
    Residual layer with multi grids
    """

    def __init__(self, n_layers, in_ch, out_ch, stride, dilation, multi_grids=None):
        super(_ResLayer, self).__init__()

        if multi_grids is None:
            multi_grids = [1 for _ in range(n_layers)]
        else:
            assert n_layers == len(multi_grids)

        # Downsampling is only in the first block
        for i in range(n_layers):
            self.add_module(
                "block{}".format(i + 1),
                _Bottleneck(
                    in_ch=(in_ch if i == 0 else out_ch),
                    out_ch=out_ch,
                    stride=(stride if i == 0 else 1),
                    dilation=dilation * multi_grids[i],
                    downsample=(True if i == 0 else False),
                ),
            )

class _ASPP(nn.Module):
    """
    Atrous spatial pyramid pooling with image-level feature
    """

    def __init__(self, in_ch, out_ch, rates):
        super(_ASPP, self).__init__()
        self.stages = nn.Module()
        self.stages.add_module("c0", _ConvBnReLU(in_ch, out_ch, 1, 1, 0, 1))
        for i, rate in enumerate(rates):
            self.stages.add_module(
                "c{}".format(i + 1),
                _ConvBnReLU(in_ch, out_ch, 3, 1, padding=rate, dilation=rate),
            )
        self.stages.add_module("imagepool", _ImagePool(in_ch, out_ch))

    def forward(self, x):
        return torch.cat([stage(x) for stage in self.stages.children()], dim=1)

class Segmentor(nn.Module):
    def __init__(self, num_classes, in_dim=[256,1024],in_size=256,pad2resize=None):
        super().__init__()
        self.in_size = in_size
        self.pad2resize = pad2resize
        bottleneck_dim = 48 # 48
        mid_dim = 256

        atrous_rates = [6, 12, 18]
        multi_grids = [1, 2, 4]
        self.aspp = _ASPP(in_dim[1], 256, atrous_rates)
        self.reduce = _ConvBnReLU(in_dim[0], bottleneck_dim, 1, 1, 0, 1)
        concat_ch = 256 * (len(atrous_rates) + 2)
        self.fc1 = _ConvBnReLU(concat_ch, mid_dim, 1, 1, 0, 1)
        self.fc2 = nn.Sequential(
            OrderedDict(
                [
                    ("conv1", _ConvBnReLU(bottleneck_dim+mid_dim, 256, 3, 1, 1, 1)),
                    ("conv2", _ConvBnReLU(256, 256, 3, 1, 1, 1)),
                    ("conv3", nn.Conv2d(256, num_classes, kernel_size=1)),
                ]
            )
        )
    
    def forward(self,x):
        # x1: (B,256,64,64) <-
        # x2: (B,512,32,32)
        # x3: (B,1024,16,16) <-
        x1,x2,x3 = x
        x1 = self.reduce(x1) # (B,256,64,64) -> (B,bottle_neck_dim,64,64)

        x3 = F.interpolate(x3, size=[self.in_size//16,self.in_size//16], mode="bilinear", align_corners=False)
        x3 = self.aspp(x3) # (B,1024,16,16) -> (B,1024,16,16)
        x3 = self.fc1(x3) # (B,1024,16,16) -> (B,256,16,16)
        x3 = F.interpolate(x3, size=[self.in_size//4,self.in_size//4], mode="bilinear", align_corners=False)
        x3 = torch.cat((x3, x1), dim=1) # (B,256,64,64) + (B,bottle_neck_dim,64,64)
        x3 = self.fc2(x3) # (B,256,64,64) -> (B,N,64,64)
        x3 = F.interpolate(x3, size=[self.in_size,self.in_size], mode="bilinear", align_corners=False)
        return x3

    