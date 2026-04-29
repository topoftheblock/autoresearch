# baseline_model.py
import torch.nn as nn
import torch.nn.functional as F

class MicroCNN(nn.Module):
    def __init__(self):
        super(MicroCNN, self).__init__()
        # Very basic, sub-optimal architecture
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(16 * 16 * 16, 10) # CIFAR images are 32x32, pooled by 2 = 16x16

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x