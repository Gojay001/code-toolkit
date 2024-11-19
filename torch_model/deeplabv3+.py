import torch
import torch.nn as nn
import torch.nn.functional as F
from aspp import build_aspp
from decoder import build_decoder
import resnet_deeplab


class DeepLab(nn.Module):
    def __init__(self, backbone='resnet', output_stride=16, num_classes=21, freeze_bn=False):
        super(DeepLab, self).__init__()
        if backbone == 'drn':
            output_stride = 8

        BatchNorm = nn.BatchNorm2d

        self.backbone = resnet_deeplab.ResNet101(BatchNorm=BatchNorm, pretrained=False, output_stride=8)
        self.aspp = build_aspp(backbone, output_stride, BatchNorm)
        self.decoder = build_decoder(num_classes, backbone, BatchNorm)

        self.freeze_bn = freeze_bn

    def forward(self, input):
        x, low_level_feat = self.backbone(input)
        x = self.aspp(x)
        x = self.decoder(x, low_level_feat)
        print(x.shape)
        x = F.interpolate(x, size=input.shape[2:], mode='bilinear', align_corners=True)

        return x

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()


if __name__ == "__main__":
    model = DeepLab(backbone='resnet', output_stride=16)
    model.eval()
    x = torch.rand(4, 3, 224, 224)
    out = model(x)
    print(out.shape)


