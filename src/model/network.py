import torch
import torch.nn as nn
import torch.nn.functional as F
from src.model.prunable_linear import PrunableLinear

class PrunableNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = PrunableLinear(32*32*3, 512)
        self.fc2 = PrunableLinear(512, 256)
        self.fc3 = PrunableLinear(256, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

    def get_all_gates(self):
        return torch.cat([
            self.fc1.get_gate_values().view(-1),
            self.fc2.get_gate_values().view(-1),
            self.fc3.get_gate_values().view(-1)
        ])
