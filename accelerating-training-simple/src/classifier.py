import torch.nn as nn
import torch as th
import torch.nn.functional as F

class Classifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 8, *(3, 1, 1))
        self.conv2 = nn.Conv2d(8, 16, *(3, 1, 1))
        self.conv3 = nn.Conv2d(16, 32, *(3, 1, 1))
        self.conv4 = nn.Conv2d(32, 64, *(3, 1, 1))

        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(64 * 2 * 2, 64)
        self.fc2 = nn.Linear(64, 100)

    def forward(self, x):
        x = self.pool(F.gelu(self.conv1(x)))
        x = self.pool(F.gelu(self.conv2(x)))
        x = self.pool(F.gelu(self.conv3(x)))
        x = self.pool(F.gelu(self.conv4(x)))
        x = th.flatten(x, 1)

        x = F.gelu(self.fc1(x))
        x = self.fc2(x)
        return x