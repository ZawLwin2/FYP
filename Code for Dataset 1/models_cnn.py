import torch.nn as nn
import torch.nn.functional as F


class EEG_CNN(nn.Module):
    def __init__(self, input_channels, num_classes):
        super(EEG_CNN, self).__init__()
        # Define the CNN layers
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * (input_channels // 2), 128)  # Adjust input size for the fully connected layer
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # Forward pass
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Example usage:
# model = EEG_CNN(input_dim, num_classes)
