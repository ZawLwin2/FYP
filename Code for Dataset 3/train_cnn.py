import torch
import torch.optim as optim
import torch.nn as nn
from sklearn.model_selection import train_test_split
from cnn_data_preparation import load_and_preprocess_data
from models_cnn import SimpleCNN


def calculate_accuracy(outputs, labels):
    _, predicted = torch.max(outputs, 1)
    correct = (predicted == labels).sum().item()
    accuracy = correct / labels.size(0)
    return accuracy


def train_model(model, train_loader, test_loader, num_epochs, learning_rate):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        running_loss = 0.0
        total_accuracy = 0.0

        # Training loop
        model.train()
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.long())
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            total_accuracy += calculate_accuracy(outputs, labels)

        avg_loss = running_loss / len(train_loader)
        avg_accuracy = total_accuracy / len(train_loader)

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.4f}")

    # Evaluate accuracy on the test set after training
    model.eval()
    with torch.no_grad():
        test_accuracy = 0.0
        total_samples = 0
        for inputs, labels in test_loader:
            outputs = model(inputs)
            test_accuracy += calculate_accuracy(outputs, labels)
            total_samples += labels.size(0)

        avg_test_accuracy = test_accuracy / len(test_loader)
        print(f"Accuracy of the model on the test set: {avg_test_accuracy * 100:.2f}%")


# Example usage:
if __name__ == "__main__":
    file_path = 'standardized_EEG_data.csv'  # Adjust the path
    batch_size = 64
    num_epochs = 20
    learning_rate = 0.001

    # Load data
    data_loader, dataset, sequence_length, num_classes = load_and_preprocess_data(file_path, batch_size)

    # Split data into train and test sets
    train_size = int(0.8 * len(dataset))  # 80% training, 20% testing
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model
    model = SimpleCNN(num_features=sequence_length, num_classes=num_classes)

    # Train the model and evaluate test accuracy
    train_model(model, train_loader, test_loader, num_epochs, learning_rate)
