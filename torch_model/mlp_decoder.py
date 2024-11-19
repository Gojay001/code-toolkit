import torch
from torch import nn
import torch.nn.functional as F


class MlpDecoder(nn.Module):
    def __init__(self, in_features, out_feature, drop_rate=0.2):
        super(MlpDecoder, self).__init__()
        self.in_features = in_features
        self.out_feature = out_feature
        self.fc1 = [nn.Linear(feature, self.out_feature) for feature in self.in_features]
        self.fc1 = nn.ModuleList(self.fc1)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(self.out_feature * 4, self.out_feature)
        self.dropout = nn.Dropout(drop_rate)
        self.fc3 = nn.Linear(self.out_feature, 2)

    def forward(self, x):
        x_list = []
        B, C, H, W = x[0].shape
        for i in range(len(self.in_features)):
            b, c, h, w = x[i].shape
            x_i = x[i].permute(0, 2, 3, 1).contiguous().view(b, -1, c)
            x_i = self.fc1[i](x_i)
            x_i = x_i.view(b, h, w, -1).permute(0, 3, 1, 2)
            x_i = F.interpolate(x_i, (H, W), mode='bilinear', align_corners=True)
            x_i = x_i.permute(0, 2, 3, 1).contiguous().view(b, H*W, -1)
            x_list.append(x_i)
        x = torch.cat(x_list, 2)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = x.view(B, H, W, -1)
        return x


if __name__ == '__main__':
    scale = [128, 256, 512, 1024]
    size = [80, 40, 20, 10]
    feat_list = [torch.randn((1, scale[i], size[i], size[i]), requires_grad=True) for i in range(4)]
    mlp = MlpDecoder(scale, 64)
    x = mlp(feat_list)
    print(x.shape)
