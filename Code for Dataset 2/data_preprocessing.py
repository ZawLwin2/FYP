import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import zscore

# Function to load, preprocess, and normalize the data
def load_and_preprocess_data(file_path):
    # Read the CSV file into a pandas DataFrame
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: The file at {file_path} was not found.")
        return None
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return None

    # Convert non-numeric columns to numeric, forcing errors to NaN
    df = df.apply(pd.to_numeric, errors='coerce')

    # Handle missing values by filling them with the mean of each column
    df.fillna(df.mean(), inplace=True)

    # Select numerical columns for normalization
    numerical_columns = df.select_dtypes(include='number').columns

    # Apply Z-score normalization to numerical columns
    df[numerical_columns] = df[numerical_columns].apply(zscore)

    # Convert DataFrame to a PyTorch tensor
    data = torch.tensor(df[numerical_columns].values, dtype=torch.float32)

    # Define the batch size
    batch_size = 64

    # Create a DataLoader for batching
    dataloader = DataLoader(TensorDataset(data), batch_size=batch_size, shuffle=True)

    return dataloader

# Define the file path for Dataset 2
file_path = r'C:\Users\zawlw\Downloads\FYP (EEG Data Augmentation)\Dataset 2\emotions.csv'

# Load, preprocess, and normalize the data
dataloader = load_and_preprocess_data(file_path)

# Check if data is loaded properly
if dataloader is not None:
    print("Data successfully loaded, preprocessed, and normalized!")

    # Iterate over the DataLoader to process data in batches
    for batch in dataloader:
        print(batch)
        # Perform your operations on each batch here, e.g., training your model
else:
    print("Data loading failed.")
