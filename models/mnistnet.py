import torch
import torch.nn as nn
from decentai.models.model_interface import ModelInterface

class Net(nn.Module, ModelInterface):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 64)
        self.fc2 = nn.Linear(64, 10)
        self.device = torch.device("cpu")

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def to(self, device):
        super().to(device)
        self.device = device
        return self