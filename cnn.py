import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    
    def forward(self, x): # x: [B, 3, 32, 32]
        x = self.pool(F.relu(self.conv1(x))) # -> [B, 6, 28, 28] -> [B, 6, 14, 14]
        x = self.pool(F.relu(self.conv2(x))) # -> [B, 16, 10, 10] -> [B, 16, 5, 5]
        x = torch.flatten(x, 1) # -> [B, 16 * 5 * 5]
        x = F.relu(self.fc1(x)) # -> [B, 120]
        x = F.relu(self.fc2(x)) # -> [B, 84]
        x = F.relu(self.fc3(x)) # -> [B, 10]
        return x
