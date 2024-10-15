import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

def load_and_preprocess_data_cnn(file_path, batch_size):
    # Load the dataset
    df = pd.read_csv(file_path)

    # Keep the relevant features; remove only the label and identifiers
    features = df.drop(columns=['SubjectID', 'VideoID', 'predefinedlabel', 'user-definedlabeln'])
    labels = df['predefinedlabel']

    # Convert features and labels to tensors
    features_tensor = torch.tensor(features.values, dtype=torch.float32)
    labels_tensor = torch.tensor(labels.values, dtype=torch.long)

    # Reshape features to [num_samples, 1, num_features] for 1D CNN
    features_tensor = features_tensor.view(features_tensor.size(0), 1, features_tensor.size(1))

    # Create a dataset
    dataset = TensorDataset(features_tensor, labels_tensor)

    return dataset  # Only return dataset
