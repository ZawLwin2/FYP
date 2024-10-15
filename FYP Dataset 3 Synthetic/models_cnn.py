import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.dropout = nn.Dropout(p=0.3)  # Lower dropout rate
        self.batch_norm = nn.BatchNorm1d(16)

        conv_output_size = input_size - 2
        pool_output_size = (conv_output_size // 2)

        self.fc1 = nn.Linear(16 * pool_output_size, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.batch_norm(self.conv1(x))))
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x
