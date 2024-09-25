from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch


class EEGDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx].unsqueeze(0), self.labels[idx]  # Add channel dimension


def load_and_preprocess_data(file_path, batch_size):
    print("Loading data...")
    df = pd.read_csv(file_path)
    print("Data loaded successfully.")

    # Print the first few rows
    print(df.head())

    # Assuming 'label' is the target column and needs to be encoded
    label_encoder = LabelEncoder()
    df['label'] = label_encoder.fit_transform(df['label'])

    # Drop the label column and split the data
    X = df.drop('label', axis=1).values
    y = df['label'].values

    # Print the shape of X to understand its dimensions
    print("Shape of X before reshaping:", X.shape)

    # Reshape X for CNN (add channel dimension)
    total_elements = X.size
    channels = 1  # example
    height = 64  # example, adjust according to your data
    width = 64  # example, adjust according to your data

    # Calculate total number of elements needed
    required_elements = channels * height * width
    if total_elements % required_elements == 0:
        X = X.reshape(-1, channels, height, width)
        print("Shape of X after reshaping:", X.shape)
    else:
        raise ValueError(f"Cannot reshape array of size {total_elements} into shape ({channels},{height},{width})")

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print("Data split into training and testing sets.")

    # Create Dataset and DataLoader
    train_dataset = EEGDataset(X_train, y_train)
    test_dataset = EEGDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
