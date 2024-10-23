import torch
import torch.nn as nn
import torch.functional as F

class MLP(nn.Module):
    def __init__(self, n_units=1000, n_in=784, n_out=10):
        super(MLP, self).__init__()
        self.l1 = nn.Linear(n_in, n_units)  # n_in -> n_units
        self.l2 = nn.Linear(n_units, n_units)  # n_units -> n_units
        self.l3 = nn.Linear(n_units, n_out)  # n_units -> n_out
        self.relu = nn.ReLU()

    def forward(self, x):
        h = self.relu(self.l1(x))
        h = self.relu(self.l2(h))
        h = self.l3(h)

        return h

class MNIST_CNN(nn.Module):
    def __init__(self, n_in=1, n_out=10):
        super(MNIST_CNN, self).__init__()
        self.conv1 = nn.Conv2d(n_in, 10, kernel_size=7, stride=1, padding=0)
        self.conv2 = nn.Conv2d(10, 10, kernel_size=5, stride=1, padding=0)
        self.conv3 = nn.Conv2d(10, 20, kernel_size=5, stride=2, padding=0)
        self.conv4 = nn.Conv2d(20, 20, kernel_size=5, stride=1, padding=0)
        # self.conv1 = '畳み込み層を定義してください'
        # self.conv2 = '畳み込み層を定義してください'
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(20, n_out)
        self.relu = nn.ReLU()

    def forward(self, x):
        h = self.relu(self.conv1(x))
        h = self.relu(self.conv2(h))
        h = self.relu(self.conv3(h))
        h = self.relu(self.conv4(h))
        h = self.avg_pool(h)
        h = torch.flatten(h, 1)
        h = self.fc(h)

        return h

class Cifar_CNN(nn.Module):
    def __init__(self, n_in=3, n_out=10):
        super(Cifar_CNN, self).__init__()
        self.conv1 = nn.Conv2d(n_in, 32, 3)
        self.maxpool2d = nn.MaxPool2d(kernel_size=3, stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.l1 = nn.Linear(32, 512)
        self.l_out = nn.Linear(512, n_out)
        self.relu = nn.ReLU()

    def forward(self, x):
        h = self.relu(self.maxpool2d(self.conv1(x)))
        h = self.avg_pool(h)
        h = torch.flatten(h, 1)
        h = self.relu(self.l1(h))
        h = self.l_out(h)

        return h


if __name__ == "__main__":
    # model = MNIST_CNN()
    model = Cifar_CNN()

    z = torch.randn(10,3,32,32)

    y = model(z)
    print(y.size())