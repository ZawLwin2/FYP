import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    def __init__(self, input_dim):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)

        # Calculate the output dimension after convolutional layers
        self.conv_output_dim = self._get_conv_output_dim(input_dim)

        self.fc1 = nn.Linear(self.conv_output_dim, 128)  # Updated to the correct size
        self.fc2 = nn.Linear(128, 1)  # Change to the number of classes if needed

    def _get_conv_output_dim(self, input_dim):
        # Create a dummy input tensor to pass through the conv layers
        x = torch.zeros(1, 1, input_dim)  # Batch size of 1, 1 channel, input_dim length
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        return x.numel()  # Total number of elements after conv layers

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x.unsqueeze(1))))  # Add channel dimension
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))  # Sigmoid for binary classification
        return x
