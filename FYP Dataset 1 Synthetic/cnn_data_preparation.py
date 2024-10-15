import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_and_preprocess_generated_data(file_path):
    # Load the generated data
    data = pd.read_csv(file_path)

    # Assuming the first column is the label (update as needed)
    X = data.values[:, :-1]  # Features
    y = data.values[:, -1]  # Target label

    # Normalize features using Z-score normalization
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test
