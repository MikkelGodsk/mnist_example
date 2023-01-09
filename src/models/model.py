import torch
from torch import nn


class MyAwesomeModel(nn.Module):
    def __init__(self, hidden_layer_neurons=28 * 28):
        super().__init__()
        # self.conv1 = nn.LazyConv1d(out_channels=3, kernel_size=5)
        # self.pool1 = nn.AvgPool2d(kernel_size=5)
        self.fc1 = nn.LazyLinear(hidden_layer_neurons, dtype=torch.float64)
        self.fc2 = nn.LazyLinear(10, dtype=torch.float64)
        self.relu = nn.ReLU()
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        if x.ndim != 3:
            raise ValueError("Input should be in 3 dimensions")
        # x = x.flatten()
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.log_softmax(x)
        return x
