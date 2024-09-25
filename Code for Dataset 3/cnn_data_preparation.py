import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler


def load_and_preprocess_data(file_path, batch_size):
    # Load dataset
    df = pd.read_csv(file_path)

    # Select the features (exclude SubjectID, VideoID, and label)
    X = df.drop(['SubjectID', 'VideoID', 'predefinedlabel'], axis=1).values

    # Select the label (predefinedlabel)
    y = df['predefinedlabel'].values

    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Convert data to tensors
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)  # Ensure labels are in long (integer) format

    # Reshape X to match CNN input expectations: (batch_size, channels, features)
    X_tensor = X_tensor.view(X_tensor.size(0), 1, -1)  # Assuming 1 channel for EEG data

    # Create dataset and dataloader
    dataset = TensorDataset(X_tensor, y_tensor)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Sequence length is the number of features after reshaping
    sequence_length = X_tensor.shape[2]
    num_classes = len(torch.unique(y_tensor))  # If y is continuous, adjust accordingly

    return data_loader, dataset, sequence_length, num_classes
