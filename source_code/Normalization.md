# Normalization

## Abstract
1. 归一化作用

- **稳定输入的分布**（零均值单位方差），
- 让每一层学习的目标保持稳定，使得**梯度更平滑**（强制使梯度处于激活函数饱和区-0值附近），
- 梯度稳定，也就能够采用更大的学习率，从而**加快收敛**。

2. 对比

`BN` 是**逐通道**的归一化操作（**NHW**），`LN` 是**逐样本/逐token**的归一化操作（**图像：CHW，序列：D**）。

**CNN**：图像具有空间不变性，相同通道特征在不同位置应有相似分布，所以可以逐通道跨样本归一化；（`BN`）

**Transformer**：`LN` 在每个样本的每个 token 内，独立地对 D 个特征进行归一化计算，让每个 token 的特征向量分布稳定；

3. 代码实现

先将数据标准化，再用两个参数进行线性缩放；(y = scale * x + shift);

Dummy implementation of `BatchNorm2d`, `LayerNorm` and `RMSNorm`:
```python
import torch
import torch.nn as nn

class BatchNorm2d:
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.gamma = nn.Parameter(torch.ones(num_features))  # 缩放参数
        self.beta = nn.Parameter(torch.zeros(num_features))  # 偏移参数
        self.running_mean = torch.zeros(num_features)       # 运行统计量-均值
        self.running_var = torch.ones(num_features)         # 运行统计量-方差
        self.training = True

    def forward(self, x):
        N, C, H, W = x.shape
        if self.training:  # 使用当前批次的统计量
            mean = x.mean(dim=(0, 2, 3), keepdim=True)
            var = x.var(dim=(0, 2, 3), keepdim=True, unbiased=False)
            with torch.no_grad():  # 更新运行统计量
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean.squeeze()
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var.squeeze()
        else:  # 使用运行统计量
            mean = self.running_mean.view(1, C, 1, 1)
            var = self.running_var.view(1, C, 1, 1)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)  # 归一化
        gamma = self.gamma.view(1, C, 1, 1)
        beta = self.beta.view(1, C, 1, 1)
        return x_norm * gamma + beta  # 仿射变换

class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(dim)) # 缩放参数
        self.beta = nn.Parameter(torch.zeros(dim)) # 偏移参数

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)          # 计算均值
        var = x.var(-1, keepdim=True)            # 计算方差（含均值中心化）
        x_norm = (x - mean) / torch.sqrt(var + self.eps) # 中心化+缩放
        return self.gamma * x_norm + self.beta   # 仿射变换

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(dim)) # 只有缩放参数

    def forward(self, x):
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) # 只计算RMS
        x_norm = x / rms                                             # 只做缩放
        return self.gamma * x_norm                                   # 仿射变换（无偏置）
```


## BatchNorm

- For **Image(NCHW)**, use channel-wise normalization(**NHW**).
- For **Sequence(NCL)**, use channel-wise normalization(**NL**).

PyTorch `nn.BatchNorm` implementation:
> https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/batchnorm.py

```python
class BatchNorm1d(_BatchNorm):
    r"""
    Shape:
        - Input: :math:`(N, C)` or :math:`(N, C, L)`, where :math:`N` is the batch size,
          :math:`C` is the number of features or channels, and :math:`L` is the sequence length
        - Output: :math:`(N, C)` or :math:`(N, C, L)` (same shape as input)

    Examples::

        >>> # With Learnable Parameters
        >>> m = nn.BatchNorm1d(100)
        >>> # Without Learnable Parameters
        >>> m = nn.BatchNorm1d(100, affine=False)
        >>> input = torch.randn(20, 100)
        >>> output = m(input)
    """

class BatchNorm2d(_BatchNorm):
    r"""
    Shape:
        - Input: :math:`(N, C, H, W)`
        - Output: :math:`(N, C, H, W)` (same shape as input)

    Examples::

        >>> # With Learnable Parameters
        >>> m = nn.BatchNorm2d(100)
        >>> # Without Learnable Parameters
        >>> m = nn.BatchNorm2d(100, affine=False)
        >>> input = torch.randn(20, 100, 35, 45)
        >>> output = m(input)
    """
```

`BatchNorm` dummy implementation:
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

class BatchNorm2d:
    def __init__(self,
                 num_features: int,
                 eps: float = 1e-5,
                 momentum: float = 0.1,
                 affine: bool = True,
                 track_running_stats: bool = True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        if affine:
            self.weight = nn.Parameter(torch.ones(num_features))  # gamma
            self.bias = nn.Parameter(torch.zeros(num_features))   # beta
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        if track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features))
            self.register_buffer('running_var', torch.ones(num_features))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_buffer('running_mean', None)
            self.register_buffer('running_var', None)
            self.register_buffer('num_batches_tracked', None)

        self.training = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() not in [2, 3, 4, 5]:
            raise ValueError(f"Expect 2-5D input (got {x.dim()}D input)")

        if x.dim() == 4:  # BatchNorm2d
            dims = (0, 2, 3)
        elif x.dim() == 3:  # BatchNorm1d
            dims = (0, 2)
        elif x.dim() == 5:  # BatchNorm3d
            dims = (0, 2, 3, 4)
        else:  # 2D
            dims = (0,)

        if self.training and self.track_running_stats:
            batch_mean = x.mean(dim=dims, keepdim=True)
            batch_var = x.var(dim=dims, keepdim=True, unbiased=False)

            with torch.no_grad():
                m = self.momentum
                self.running_mean = (1 - m) * self.running_mean + m * batch_mean.squeeze()
                self.running_var = (1 - m) * self.running_var + m * batch_var.squeeze()
                self.num_batches_tracked += 1

            mean = batch_mean
            var = batch_var

        elif not self.training and self.track_running_stats:
            shape = [1] * x.dim()
            shape[1] = -1
            mean = self.running_mean.view(*shape)
            var = self.running_var.view(*shape)

        else:
            mean = x.mean(dim=dims, keepdim=True)
            var = x.var(dim=dims, keepdim=True, unbiased=False)

        normalized = (x - mean) / torch.sqrt(var + self.eps)

        if self.affine and self.weight is not None and self.bias is not None:
            weight_shape = [1] * x.dim()
            weight_shape[1] = -1
            weight = self.weight.view(*weight_shape)
            bias = self.bias.view(*weight_shape)

            normalized = normalized * weight + bias

        return normalized
```


## LayerNorm

- For **Image(NCHW)**, use sample-wise normalization, that normalize over the last three dimensions(**CHW**).
- For **Sequence(NLD)**, use token-wise normalization, that only normalize over the last dimension(**D**).

`nn.LayerNorm`是一个包装类，主要负责管理可学习的仿射参数（weight和 bias），而实际的归一化计算则委托给底层的 `torch.nn.functional.layer_norm`函数（或更底层的 C++ 实现）完成。

### nn.LayerNorm
PyTorch implementation:
> [https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/normalization.py#L105](https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/normalization.py#L105)

```python
class LayerNorm(Module):
    r"""
    Examples::

        >>> # NLP Example
        >>> batch, sentence_length, embedding_dim = 20, 5, 10
        >>> embedding = torch.randn(batch, sentence_length, embedding_dim)
        >>> layer_norm = nn.LayerNorm(embedding_dim)
        >>> # Activate module
        >>> output = layer_norm(embedding)
        >>>
        >>> # Image Example
        >>> N, C, H, W = 20, 5, 10, 10
        >>> input = torch.randn(N, C, H, W)
        >>> # Normalize over the last three dimensions (i.e. the channel and spatial dimensions)
        >>> # as shown in the image below
        >>> layer_norm = nn.LayerNorm([C, H, W])
        >>> output = layer_norm(input)
    """
```

### F.layer_norm
PyTorch interface:
> [https://github.com/pytorch/pytorch/blob/main/torch/nn/functional.py#L2924](https://github.com/pytorch/pytorch/blob/main/torch/nn/functional.py#L2924)

C++ implementation:
> [https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cpu/layer_norm_kernel.cpp#L59](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cpu/layer_norm_kernel.cpp#L59)

CUDA implementation:
> [https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/layer_norm_kernel.cu#L253](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/layer_norm_kernel.cu#L253)

PyTorch dummy implementation:
```python
def layer_norm(input, normalized_shape, weight=None, bias=None, eps=1e-5):
    # select dims from end to len(normalized_shape)
    dims = list(range(input.dim() - len(normalized_shape), input.dim()))

    mean = input.mean(dim=dims, keepdim=True)
    var = input.var(dim=dims, keepdim=True, unbiased=False)

    normalized = (input - mean) / torch.sqrt(var + eps)

    if weight is not None and bias is not None:
        normalized = normalized * weight + bias
    elif weight is not None:
        normalized = normalized * weight
    elif bias is not None:
        normalized = normalized + bias

    return normalized
```


## RMSNorm

去除了减去均值（中心化）的操作，只保留除以均方根（缩放）的部分，同时去除了仿射变换的偏置 β；

中心化和偏置可以通过 FFN 补偿，而缩放能够稳定梯度范围，才是LN的关键；
> y = W * (x - μ) + b = W * x - W * μ + b = W * x + (b - W * μ);

### nn.RMSNorm
PyTorch implementation:
> https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/normalization.py#L343
```python
class RMSNorm(Module):
    r"""
    Examples::

        >>> rms_norm = nn.RMSNorm([2, 3])
        >>> input = torch.randn(2, 2, 3)
        >>> output = rms_norm(input)
    """
```
