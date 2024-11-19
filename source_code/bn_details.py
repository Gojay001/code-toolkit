import torch
import torch.nn as nn
import torch.nn.modules.batchnorm


def create_inputs():
    return torch.randn(8, 3, 20, 20)

# 以 BatchNorm2d 为例实现 forward
# mean_val, var_val 不为None时，不对输入进行统计，而直接用传进来的均值、方差
def dummy_bn_forward(x, bn_weight, bn_bias, eps, mean_val=None, var_val=None):
    if mean_val is None:
        mean_val = x.mean([0, 2, 3])
    if var_val is None:
        # 这里需要注意，torch.var 默认算无偏估计，因此需要手动设置unbiased=False
        var_val = x.var([0, 2, 3], unbiased=False)

    x = x - mean_val[None, ..., None, None]
    x = x / torch.sqrt(var_val[None, ..., None, None] + eps)
    x = x * bn_weight[..., None, None] + bn_bias[..., None, None]
    return mean_val, var_val, x

# 验证 dummy BN 输出
def check_bn_forward(in_channel=3):
    bn_layer = nn.BatchNorm2d(num_features=in_channel)
    inputs = create_inputs()
    bn_outputs = bn_layer(inputs)  # 用 pytorch 的实现 forward
    # 用 dummy_bn 实现 forward
    _, _, dummy_outputs = dummy_bn_forward(
        inputs, bn_layer.weight, bn_layer.bias, bn_layer.eps)
    assert torch.allclose(dummy_outputs, bn_outputs)
    print('dummy_ouputs is closed to bn_ouputs.')

# 验证 mean, var 的更新
def check_mv_update(in_channel=3, momentum=0.1):
    running_mean = torch.zeros(in_channel)
    running_var = torch.ones_like(running_mean)
    num_batches_tracked = torch.tensor(0, dtype=torch.long)
    bn_layer = nn.BatchNorm2d(num_features=3, momentum=momentum)

    # 模拟 forward 10 次
    for t in range(10):
        inputs = create_inputs()
        bn_outputs = bn_layer(inputs)
        inputs_mean, inputs_var, _ = dummy_bn_forward(
            inputs, bn_layer.weight, bn_layer.bias, bn_layer.eps
        )
        num_batches_tracked += 1
        eaf = momentum if momentum else 1.0 / float(num_batches_tracked)  # exponential_average_factor
        n = inputs.numel() / inputs.size(1)
        # 更新 running_var 和 running_mean
        running_var = running_var * (1 - eaf) + eaf * inputs_var * n / (n - 1)
        running_mean = running_mean * (1 - eaf) + eaf * inputs_mean

    assert torch.allclose(running_var, bn_layer.running_var)
    assert torch.allclose(running_mean, bn_layer.running_mean)

    # check inference outputs
    bn_layer.train(mode=False)
    inference_inputs = create_inputs()
    bn_outputs = bn_layer(inference_inputs)
    _, _, dummy_outputs = dummy_bn_forward(
        inference_inputs, bn_layer.weight,
        bn_layer.bias, bn_layer.eps,
        running_mean, running_var)
    assert torch.allclose(dummy_outputs, bn_outputs)
    print('inference: dummy_ouputs is closed to bn_ouputs.')
    print(f'bn_layer running_mean is {bn_layer.running_mean}')
    print(f'dummy bn running_mean is {running_mean}')
    print(f'bn_layer running_var is {bn_layer.running_var}')
    print(f'dummy bn running_var is {running_var}')

# 验证 \gamma, \beta 的更新
def check_gb_update():
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader
    import torch.nn.functional as F

    # 用 mnist 作为 toy dataset
    dataset = datasets.MNIST(root='../data', download=True, transform=transforms.ToTensor())
    dataloader = DataLoader(dataset=dataset, batch_size=8)

    # 初始化一个带 BN 的简单模型
    toy_model = nn.Sequential(nn.Linear(28 ** 2, 128), nn.BatchNorm1d(128),
                              nn.ReLU(), nn.Linear(128, 10), nn.Sigmoid())
    optimizer = torch.optim.SGD(toy_model.parameters(), lr=0.1)

    bn_1d_layer = toy_model[1]
    print(f'Initial weight is {bn_1d_layer.weight[:4].tolist()}...')
    print(f'Initial bias is {bn_1d_layer.bias[:4].tolist()}...\n')
    # 模拟更新2次参数
    for (i, data) in enumerate(dataloader):
        output = toy_model(data[0].view(data[0].shape[0], -1))
        loss = F.cross_entropy(output, data[1])
        loss.backward()
        # 输出部分参数的梯度，验证weight和bias确实是通过gradient descent更新的
        print(f'The {i+1}th Gradient of weight is {bn_1d_layer.weight.grad[:4].tolist()}...')
        print(f'The {i+1}th Gradient of bias is {bn_1d_layer.bias.grad[:4].tolist()}...')
        optimizer.step()
        optimizer.zero_grad()
        if i == 1:
            break
    print(f'\nNow weight is {bn_1d_layer.weight[:4].tolist()}...')
    print(f'Now bias is {bn_1d_layer.bias[:4].tolist()}...')

    inputs = torch.randn(4, 128)
    bn_outputs = bn_1d_layer(inputs)
    new_bn = nn.BatchNorm1d(128)
    bn_outputs_no_weight_bias = new_bn(inputs)

    assert not torch.allclose(bn_outputs, bn_outputs_no_weight_bias)

# 验证 eval 模式下 bn 的输出
def check_bn_eval(in_channel=3):
    inference_inputs = create_inputs()
    bn_layer = nn.BatchNorm2d(num_features=in_channel)

    bn_layer.train(mode=True)
    bn_outputs = bn_layer(inference_inputs)
    running_mean = bn_layer.running_mean
    running_var = bn_layer.running_var

    inference_inputs = create_inputs()
    bn_layer.train(mode=False)
    # 输出前后的 running_mean 和 running_var，验证eval模式下不再更新统计量
    print(f'bn_layer running_mean is {bn_layer.running_mean}')
    print(f'bn_layer running_var is {bn_layer.running_var}')
    bn_outputs = bn_layer(inference_inputs)
    print(f'Now bn_layer running_mean is {bn_layer.running_mean}')
    print(f'Now bn_layer running_var is {bn_layer.running_var}')
    # 用之前统计的running_mean和running_var替代输入的running_mean和running_var
    _, _, dummy_outputs = dummy_bn_forward(
        inference_inputs, bn_layer.weight,
        bn_layer.bias, bn_layer.eps,
        running_mean, running_var)
    assert torch.allclose(dummy_outputs, bn_outputs)

    # 关闭track_running_stats后，即使在eval模式下，也会去计算输入的mean和var
    bn_layer = nn.BatchNorm2d(num_features=in_channel)
    bn_layer.track_running_stats = False
    bn_outputs_notrack = bn_layer(inference_inputs)
    _, _, dummy_outputs_notrack = dummy_bn_forward(
        inference_inputs, bn_layer.weight,
        bn_layer.bias, bn_layer.eps)

    assert torch.allclose(dummy_outputs_notrack, bn_outputs_notrack)
    assert not torch.allclose(bn_outputs, bn_outputs_notrack)

if __name__ == '__main__':
    check_bn_forward()
    check_mv_update()
    check_gb_update()
    check_bn_eval()