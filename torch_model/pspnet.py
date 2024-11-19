import torch
import torch.nn as nn
import torch.nn.functional as F
import resnet
from collections import OrderedDict


class ConvBnRelu(nn.Module):
    def __init__(self, in_planes, out_planes, size, stride, pad, dilation=1,
                 groups=1, has_bn=True, norm_layer=nn.BatchNorm2d, bn_eps=1e-5,
                 has_relu=True, inplace=True, has_bias=False):
        super(ConvBnRelu, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=size,
                              stride=stride, padding=pad,
                              dilation=dilation, groups=groups, bias=has_bias)
        self.has_bn = has_bn
        if self.has_bn:
            self.bn = norm_layer(out_planes, eps=bn_eps)
        self.has_relu = has_relu
        if self.has_relu:
            self.relu = nn.ReLU(inplace=inplace)

    def forward(self, x):
        x = self.conv(x)
        if self.has_bn:
            x = self.bn(x)
        if self.has_relu:
            x = self.relu(x)

        return x


class PyramidPooling(nn.Module):
    def __init__(self, name, out_planes, fc_dim=2048, pool_scales=[1, 2, 3, 6],
                 norm_layer=nn.BatchNorm2d):
        super(PyramidPooling, self).__init__()

        self.ppm = []
        for scale in pool_scales:
            self.ppm.append(nn.Sequential(OrderedDict([
                ('{}/pool_1'.format(name), nn.AdaptiveAvgPool2d(scale)),
                ('{}/cbr'.format(name),
                 ConvBnRelu(fc_dim, 512, 1, 1, 0, has_bn=True,
                            has_relu=True, has_bias=False,
                            norm_layer=norm_layer))
            ])))
        self.ppm = nn.ModuleList(self.ppm)

    def forward(self, x):
        _, _, h, w = x.shape
        ppm_out = [x]
        for pooling in self.ppm:
            x_pool = pooling(x)
            x_ppm = F.interpolate(x_pool, size=(h, w), mode='bilinear', align_corners=True)
            ppm_out.append(x_ppm)
        ppm_out = torch.cat(ppm_out, 1)

        return ppm_out


class PSPNet(nn.Module):
    def __init__(self, class_num, bn_momentum=0.01, pretrained=False):
        super(PSPNet, self).__init__()
        self.pool_scales = [1, 2, 3, 6]
        self.fc_dim = 2048
        resnet101 = resnet.resnet101(pretrained=pretrained)
        self.backbone = nn.Sequential(*list(resnet101.children())[:-2])
        self.psp_layer = PyramidPooling('psp', out_planes=class_num, fc_dim=self.fc_dim,
                                        pool_scales=self.pool_scales, norm_layer=nn.BatchNorm2d)
        self.cls_head = nn.Sequential(
            ConvBnRelu(self.fc_dim + len(self.pool_scales) * 512, 512, size=3, stride=1, pad=1,
                       has_bn=True, has_relu=True, has_bias=False, norm_layer=nn.BatchNorm2d),
            nn.Dropout2d(0.1, inplace=False),
            nn.Conv2d(512, class_num, kernel_size=1)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        x = self.backbone(x)
        x_psp = self.psp_layer(x)
        out = self.cls_head(x_psp)
        out = F.interpolate(out, size=(h, w), mode='bilinear', align_corners=True)

        return out


if __name__ == '__main__':
    num_classes = 10
    x = torch.randn(4, 3, 224, 224)
    net = PSPNet(class_num=num_classes)
    print(net)
    out = net(x)
    print(out.shape)
