import torch
import torch.nn as nn
import torch.optim as optim
from cnn_data_preparation import load_and_preprocess_data
from models_cnn import EEG_CNN  # Import your model

# Define the path to your dataset
file_path = r'C:\Users\zawlw\Downloads\FYP (EEG Data Augmentation)\Dataset 2\standardized_mental_state.csv'

# Define batch size
batch_size = 64

# Load data
result = load_and_preprocess_data(file_path, batch_size)

# Check if the function returned the expected values
if result is None or len(result) != 4:
    raise ValueError("load_and_preprocess_data_cnn did not return the expected number of values.")

train_loader, test_loader, sequence_length, num_classes = result

# Check if data is loaded correctly
print(f"Sequence length: {sequence_length}")
print(f"Number of classes: {num_classes}")

# Initialize the model, loss function, and optimizer
input_channels = 1  # Since data has 1 channel after reshaping
model = EEG_CNN(input_channels=input_channels, num_classes=num_classes, sequence_length=sequence_length)
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
