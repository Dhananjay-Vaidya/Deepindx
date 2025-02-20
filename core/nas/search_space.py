import torch.nn as nn
import torch.nn.functional as F

class SearchSpace(nn.Module):
    def __init__(self, num_classes=10):
        super(SearchSpace, self).__init__()
        
        # Define multiple possible architectures in the search space
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(128 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)  # Reduce size by half
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)  # Reduce size again
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
