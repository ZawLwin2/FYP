import torch
import torch.nn as nn
import torch.optim as optim
from cnn_data_preparation import load_and_preprocess_data  # Import the new function

# Define the path to your dataset
file_path = r'C:\Users\zawlw\Downloads\FYP (EEG Data Augmentation)\Dataset 1\standardized_mental_state.csv'  # Replace with your file path

# Load data
train_loader, test_loader, input_dim, num_classes =load_and_preprocess_data(file_path)

# Check if data is loaded correctly
print(f"Input dimension: {input_dim}")
print(f"Number of classes: {num_classes}")

# Define your CNN model
class CNNModel(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(CNNModel, self).__init__()
        self.network = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(32 * (input_dim // 4), num_classes)  # Adjust the size based on input_dim
        )

    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension for 1D convolution
        return self.network(x)

# Initialize the model, loss function, and optimizer
model = CNNModel(input_dim, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# Testing loop
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Accuracy of the model on the test set: {accuracy:.2f}%")
