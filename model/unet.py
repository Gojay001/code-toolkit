import torch
import torch.nn as nn


def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, (3, 3)),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, (3, 3)),
        nn.ReLU(inplace=True)
    )


class UNet(nn.Module):
    def __init__(self, n_class):
        super(UNet, self).__init__()
        self.dconv_down1 = double_conv(3, 64)
        self.dconv_down2 = double_conv(64, 128)
        self.dconv_down3 = double_conv(128, 256)
        self.dconv_down4 = double_conv(256, 512)
        self.dconv5 = double_conv(512, 1024)

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.dconv_up4 = double_conv(512 + 1024, 512)
        self.dconv_up3 = double_conv(256 + 512, 256)
        self.dconv_up2 = double_conv(128 + 256, 128)
        self.dconv_up1 = double_conv(128 + 64, 64)
        
        self.conv_last = nn.Conv2d(64, n_class, (1, 1))

    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)
        
        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)   
        
        conv4 = self.dconv_down4(x)
        x = self.maxpool(conv4)

        x = self.dconv5(x)

        x = self.upsample(x)
        x = torch.cat([x, conv4[:, :, 4: (4 + x.shape[2]), 4: (4 + x.shape[3])]], dim=1)
        x = self.dconv_up4(x)

        x = self.upsample(x)
        x = torch.cat([x, conv3[:, :, 16: (16 + x.shape[2]), 16: (16 + x.shape[3])]], dim=1)
        x = self.dconv_up3(x)

        x = self.upsample(x)        
        x = torch.cat([x, conv2[:, :, 40: (40 + x.shape[2]), 40: (40 + x.shape[3])]], dim=1)
        x = self.dconv_up2(x)

        x = self.upsample(x)        
        x = torch.cat([x, conv1[:, :, 88: (88 + x.shape[2]), 88: (88 + x.shape[3])]], dim=1)
        x = self.dconv_up1(x)
        
        out = self.conv_last(x)
        
        return out


if __name__ == '__main__':
    x = torch.rand((4, 3, 572, 572), requires_grad=True)
    net = UNet(2)
    print(net)
    out = net(x)
    print(out.shape)
