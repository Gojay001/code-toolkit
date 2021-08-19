import time
import torch
import torchvision
import torch.nn as nn
import sys
import torchvision.transforms as transforms
from torch.utils.data.dataloader import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def batch_norm(is_training, X, gamma, beta, moving_mean, moving_var, eps, momentum):
    if not is_training:
        X_hat = (X - moving_mean) / torch.sqrt(moving_var + eps)
    else:
        assert len(X.shape) in (2, 4)
        if len(X.shape) == 2:
            # [N,D] -> mean N dim
            mean = X.mean(dim=0)
            var = ((X - mean) ** 2).mean(dim=0)
        else:
            # [N,C,H,W] -> mean NHW dims
            # keep X dim to broadcast computing
            mean = X.mean(dim=0, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
            var = ((X - mean) ** 2).mean(dim=0, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
        X_hat = (X - mean) / torch.sqrt(var + eps)

        moving_mean = momentum * moving_mean + (1.0 - momentum) * mean
        moving_var = momentum * moving_var + (1.0 - momentum) * var
    Y = gamma * X_hat + beta  # scale and shift
    return Y, moving_mean, moving_var


class BatchNorm(nn.Module):
    def __init__(self, num_features, num_dims):
        super(BatchNorm, self).__init__()
        if num_dims == 2:
            shape = (1, num_features)
        else:
            shape = (1, num_features, 1, 1)
        # init: scale -> 1, shift -> 0
        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))
        self.moving_mean = torch.zeros(shape)
        self.moving_var = torch.zeros(shape)

    def forward(self, X):
        if self.moving_mean.device != X.device:
            self.moving_mean = self.moving_mean.to(X.device)
            self.moving_var = self.moving_var.to(X.device)
        # save updated moving_mean and moving_var
        Y, self.moving_mean, self.moving_var = batch_norm(self.training,
                                                          X, self.gamma, self.beta, self.moving_mean,
                                                          self.moving_var, eps=1e-5, momentum=0.9)
        return Y


class FlattenLayer(nn.Module):
    def __init__(self):
        super(FlattenLayer, self).__init__()

    def forward(self, x):  # x shape: (batch, *, *, ...)
        return x.view(x.shape[0], -1)


net = nn.Sequential(
    nn.Conv2d(1, 6, kernel_size=5),
    BatchNorm(6, num_dims=4),
    nn.Sigmoid(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Conv2d(6, 16, kernel_size=5),
    BatchNorm(16, num_dims=4),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(kernel_size=2, stride=2),
    FlattenLayer(),
    nn.Linear(16 * 4 * 4, 120),
    BatchNorm(120, num_dims=2),
    nn.Sigmoid(),
    nn.Linear(120, 84),
    BatchNorm(84, num_dims=2),
    nn.Sigmoid(),
    nn.Linear(84, 10)
)
net = net.to(device)

# get Data
batch_size = 256
transform = transforms.Compose([transforms.ToTensor()])
train_set = torchvision.datasets.FashionMNIST(root='~/Datasets/FashionMNIST',
                                              train=True, transform=transform)
test_set = torchvision.datasets.FashionMNIST(root='~/Datasets/FashionMNIST',
                                             train=False, transform=transform)
train_iter = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
test_iter = DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=0)

lr, num_epochs = 0.001, 5
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=lr)


# evaluate_accuracy
def evaluate_accuracy(test_iterator, net):
    with torch.no_grad():
        device = list(net.parameters())[0].device
        test_acc_sum = 0.0
        ncount = 0
        for x_test, y_test in test_iterator:
            if isinstance(net, torch.nn.Module):
                net.eval()
                x_test = x_test.to(device)
                y_test = y_test.to(device)
                y_hat = net(x_test)
                test_acc_sum += (y_hat.argmax(dim=1) == y_test).sum().cpu().item()
                ncount += len(y_test)
                net.train()
        test_acc = test_acc_sum / ncount
        return test_acc


def train(num_epoch):
    for epoch in range(num_epoch):
        l_sum, train_acc_sum, ncount, start = 0.0, 0.0, 0, time.time()
        for x_train, y_train in train_iter:
            x_train = x_train.to(device)
            y_train = y_train.to(device)
            y_hat = net(x_train)
            l = loss(y_hat, y_train)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            l_sum += l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y_train).sum().cpu().item()
            ncount += y_train.shape[0]
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch: %d, train_loss: %.4f, train_acc: %.4f, test_acc: %.4f , spend_time: %.4f' %
              (epoch + 1, l_sum / ncount, train_acc_sum / ncount, test_acc, time.time() - start))


if __name__ == "__main__":
    train(num_epoch=5)
