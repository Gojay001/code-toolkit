import torch
from torch import nn
from torchvision import models


# This is implemented in full accordance with the original one (caffe)
# (https://github.com/shelhamer/fcn.berkeleyvision.org)
class FCN8s(nn.Module):
    def __init__(self, num_classes, pretrained=False):
        super(FCN8s, self).__init__()
        if pretrained:
            vgg = models.vgg16(pretrained=True)
        else:
            vgg = models.vgg16(pretrained=False)

        features, classifier = list(vgg.features.children()), list(vgg.classifier.children())

        '''
        100 padding for 2 reasons:
            1) support very small input size
            2) allow cropping in order to match size of different layers' feature maps
        Note that the cropped part corresponds to a part of the 100 padding
        Spatial information of different layers' feature maps cannot be align exactly because of cropping, which is bad
        '''
        features[0].padding = (100, 100)

        self.features3 = nn.Sequential(*features[: 17])
        self.features4 = nn.Sequential(*features[17: 24])
        self.features5 = nn.Sequential(*features[24:])

        self.score_pool3 = nn.Conv2d(256, num_classes, kernel_size=1)
        self.score_pool4 = nn.Conv2d(512, num_classes, kernel_size=1)

        fc6 = nn.Conv2d(512, 4096, kernel_size=7)
        fc7 = nn.Conv2d(4096, 4096, kernel_size=1)
        score_fr = nn.Conv2d(4096, num_classes, kernel_size=1)

        self.score_fr = nn.Sequential(
            fc6, nn.ReLU(inplace=True), nn.Dropout(), fc7, nn.ReLU(inplace=True), nn.Dropout(), score_fr
        )

        self.upscore2 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=4, stride=2, bias=False)
        self.upscore_pool4 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=4, stride=2, bias=False)
        self.upscore8 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=16, stride=8, bias=False)

    def forward(self, x):
        _, _, H, W = x.shape
        pool3 = self.features3(x)      # [B, 256, 52, 52]
        pool4 = self.features4(pool3)  # [B, 512, 26, 26]
        pool5 = self.features5(pool4)  # [B, 512, 13, 13]

        score_fr = self.score_fr(pool5)     # [B, cls, 7, 7]
        upscore2 = self.upscore2(score_fr)  # [B, cls, 16, 16]

        score_pool4 = self.score_pool4(0.01 * pool4)    # [B, cls, 26, 26]
        upscore_pool4 = self.upscore_pool4(score_pool4[:, :, 5: (5 + upscore2.shape[2]), 5: (5 + upscore2.shape[3])]
                                           + upscore2)  # [B, cls, 16, 16] -> [B, cls, 34, 34]

        score_pool3 = self.score_pool3(0.0001 * pool3)  # [B, cls, 52, 52]
        upscore8 = self.upscore8(score_pool3[:, :, 9: (9 + upscore_pool4.shape[2]), 9: (9 + upscore_pool4.shape[3])]
                                 + upscore_pool4)       # [B, cls, 34, 34] -> [B, cls, 280, 280]
        return upscore8[:, :, 31: (31 + H), 31: (31 + W)].contiguous()


if __name__ == '__main__':
    num_classes = 10
    x = torch.randn((4, 3, 224, 224), requires_grad=True)
    net = FCN8s(num_classes)
    out = net(x)
    print(out.shape)
