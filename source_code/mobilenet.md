# MobileNet

> [TODO]: Update notes.

Traditional conv: `conv + BN + ReLU`;

MobileNetV1(DepthwiseSeparableConv):

- depthwise: conv3x3 + BN + ReLU6;
- pointwise: conv1x1 + BN + ReLU6;

MobileNetV2(ConvDepthwiseConv / InvertedResidual):

- pointwise: conv1x1 + BN + ReLU6;
- depthwise: conv3x3 + BN +ReLU6;
- pointwise: conv1x1 + BN;
- residual: x += f(x);

## Implementation

> timm implementation can be found in: [timm-InvertedResidual](https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/_efficientnet_blocks.py#L234).

```python
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.depthwise = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels),
            nn.BatchNorm2d(in_channels),
            nn.ReLU6())
        self.pointwise = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6())

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class ConvDepthwiseConv(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, **unused):
        super(ConvDepthwiseConv, self).__init__()
        self.use_residual = stride == 1 and in_channels == out_channels
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU6())
        self.depthwise = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, kernel_size, stride, padding, groups=mid_channels, dilation=dilation),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU6())
        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels))

    def forward(self, x):
        out = self.conv1(x)
        out = self.depthwise(out)
        out = self.conv2(out)
        if self.use_residual:
            return x + out
        else:
            return out
```

