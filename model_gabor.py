import torch
import torch.nn as nn
import torch.nn.functional as F
from gabor_filters import generate_gabor_filters

class GaborCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)

        # Replace conv1's random-initialized weights with Gabor filters
        gabor_weights = generate_gabor_filters(num_orientations=8, num_scales=4, ksize=3)
        self.conv1.weight = nn.Parameter(gabor_weights, requires_grad=False)  # frozen

        # Everything after this is identical to SimpleCNN, and stays trainable
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


if __name__ == "__main__":
    model = GaborCNN()
    dummy_input = torch.randn(64, 3, 32, 32)
    output = model(dummy_input)
    print("Output shape:", output.shape)
    print("conv1 requires_grad:", model.conv1.weight.requires_grad)  # should print False