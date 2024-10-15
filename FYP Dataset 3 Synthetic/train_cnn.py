import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
from cnn_data_preparation import load_and_preprocess_data_cnn
from models_cnn import SimpleCNN  # Import your CNN model class

# File path for your dataset
file_path = r'C:\Users\zawlw\Downloads\FYP (EEG Data Augmentation)\Dataset 3\synthetic_data.csv'

# Load your dataset to determine num_classes
data = pd.read_csv(file_path)

num_classes = data['predefinedlabel'].nunique()  # Get number of unique labels

# Drop unnecessary columns
features_to_drop = ['SubjectID', 'VideoID', 'predefinedlabel', 'user-definedlabeln']
remaining_data = data.drop(columns=features_to_drop, errors='ignore')
input_size = remaining_data.shape[1]  # Set input size based on remaining features

# Hyperparameters
batch_size = 64
num_epochs = 20
learning_rate = 0.0001  # Adjusted learning rate
patience = 5  # Early stopping patience
k_folds = 5  # Number of folds for cross-validation

# Load and preprocess data
dataset = load_and_preprocess_data_cnn(file_path, batch_size)

# Initialize KFold
kf = KFold(n_splits=k_folds)

# K-Fold Cross Validation
for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
    print(f'Fold {fold + 1}/{k_folds}')

    # Create data loaders for this fold
    train_subset = Subset(dataset, train_idx)
    val_subset = Subset(dataset, val_idx)
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

    # Initialize the model, loss function, and optimizer
    model = SimpleCNN(input_size, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.0001)

    # Early stopping parameters
    best_val_loss = float('inf')
    patience_counter = 0

    # Training loop
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        for inputs, labels in train_loader:
            optimizer.zero_grad()  # Zero the gradients
            outputs = model(inputs)  # Forward pass
            loss = criterion(outputs, labels)  # Compute loss
            loss.backward()  # Backward pass
            optimizer.step()  # Update weights

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_predictions += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

        # Calculate average loss and accuracy for this epoch
        avg_loss = running_loss / len(train_loader)
        accuracy = (correct_predictions / total_predictions) * 100

        # Print loss and accuracy for this epoch
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')

        # Validation logic
        model.eval()  # Set the model to evaluation mode
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, val_predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (val_predicted == labels).sum().item()

        val_loss /= len(val_loader)
        val_accuracy = (val_correct / val_total) * 100
        print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%')

        # Check for improvement in validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), f'cnn_model_fold{fold + 1}.pth')  # Save the best model for this fold
            print(f"Model improved and saved as cnn_model_fold{fold + 1}.pth")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered. No improvement in validation loss.")
                break

# Final model save
print("Training completed for all folds.")
