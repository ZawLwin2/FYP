import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    def __init__(self, num_features, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        # Calculate the size of the input to the fully connected layer
        # num_features is the sequence length of the input, // 4 accounts for two max-pooling layers
        fc_input_size = 32 * (num_features // 4)

        self.fc1 = nn.Linear(fc_input_size, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # First conv + pooling
        x = self.pool(F.relu(self.conv2(x)))  # Second conv + pooling

        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Example validation:
if __name__ == "__main__":
    num_features = 14  # Example, adjust based on your data
    num_classes = 2  # Example, adjust based on your data
    model = SimpleCNN(num_features=num_features, num_classes=num_classes)
    print(model)
