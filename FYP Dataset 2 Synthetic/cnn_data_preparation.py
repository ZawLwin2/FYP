import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch

class EEGDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]  # No need to unsqueeze for 1D CNN
def load_and_preprocess_data(file_path, batch_size):
    print("Loading data...")
    df = pd.read_csv(file_path, header=None)  # Set header=None if your CSV has no header
    print("Data loaded successfully.")

    # Print the shape of DataFrame to understand its dimensions
    print("Shape of DataFrame:", df.shape)
    print("Initial data preview:")
    print(df.head())

    # Convert DataFrame to numeric values, forcing non-numeric to NaN
    df = df.apply(pd.to_numeric, errors='coerce')

    # Check for any NaN values and handle them
    if df.isnull().values.any():
        print("Warning: Found NaN values in the data. Filling NaN with 0.")
        df = df.fillna(0)  # You can also choose to drop NaN values with df.dropna()

    # Assuming there are no labels, treat the entire dataset as features
    X = df.values

    # Print the shape of X to understand its dimensions
    print("Shape of X before reshaping:", X.shape)

    # Reshape X for Conv1d
    num_samples, num_features = X.shape
    channels = 1  # Example, adjust as needed
    sequence_length = num_features  # For 1D CNN, sequence length is the number of features

    # Reshape X
    X = X.reshape(num_samples, channels, sequence_length)
    print("Shape of X after reshaping:", X.shape)

    # Split into training and testing sets
    X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)
    print("Data split into training and testing sets.")

    # Create Dataset and DataLoader with dummy labels (zeros)
    train_dataset = EEGDataset(X_train, np.zeros(len(X_train)))  # Dummy labels
    test_dataset = EEGDataset(X_test, np.zeros(len(X_test)))  # Dummy labels

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Return loaders, sequence_length, and a dummy number of classes (e.g., 1)
    return train_loader, test_loader, sequence_length, 1
