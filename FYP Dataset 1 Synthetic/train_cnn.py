import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import torch.nn as nn
import numpy as np
from cnn_data_preparation import load_and_preprocess_generated_data
from models_cnn import SimpleCNN
from sklearn.metrics import accuracy_score

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load and preprocess the generated data
file_path = r'C:\Users\zawlw\Downloads\FYP (EEG Data Augmentation)\Dataset 1\synthetic_data.csv'  # Replace with your generated data file path
X_train, X_test, y_train, y_test = load_and_preprocess_generated_data(file_path)

# Convert data to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test = torch.tensor(y_test, dtype=torch.float32).to(device)

# Convert y_train and y_test to binary labels if necessary
y_train = np.where(y_train > 0, 1, 0)  # Adjust based on your specific label criteria
y_test = np.where(y_test > 0, 1, 0)  # Adjust as well

# Convert to PyTorch tensors
y_train = torch.tensor(y_train, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# Create DataLoader
train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), y_train)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Initialize model, loss function, and optimizer
input_dim = X_train.size(1)  # Number of input features
model = SimpleCNN(input_dim).to(device)
criterion = nn.BCELoss()  # Use CrossEntropyLoss for multi-class
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 20  # Number of training epochs
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()

    # Forward pass
    outputs = model(X_train)
    loss = criterion(outputs.view(-1), y_train)

    # Backward pass and optimization
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 1 == 0:  # Print every 5 epochs
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

# Evaluation
model.eval()  # Set the model to evaluation mode
with torch.no_grad():
    outputs = model(torch.tensor(X_test, dtype=torch.float32))  # Forward pass on test set
    outputs = outputs.view(-1)  # Flatten if necessary
    y_pred_labels = (outputs > 0.5).float()  # Convert probabilities to binary labels

# Calculate accuracy
accuracy = accuracy_score(y_test.cpu(), y_pred_labels.cpu())  # Ensure both are on CPU for accuracy calculation
print(f'Accuracy: {accuracy * 100:.2f}%')
