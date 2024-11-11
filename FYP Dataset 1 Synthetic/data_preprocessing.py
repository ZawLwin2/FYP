import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset


# Function to load and preprocess the data
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

    # Convert DataFrame to a PyTorch tensor
    data = torch.tensor(df.values, dtype=torch.float32)

    # Define the batch size
    batch_size = 64

    # Create a DataLoader for batching
    dataloader = DataLoader(TensorDataset(data), batch_size=batch_size, shuffle=True)

    return dataloader


# Define the file path
file_path = r'C:\Users\zawlw\Downloads\FYP (EEG Data Augmentation)\Dataset 1\standardized_mental_state (dataset 1).csv'

# Load the data
dataloader = load_and_preprocess_data(file_path)

# Check if data is loaded properly
if dataloader is not None:
    print("Data successfully loaded and preprocessed!")

    # Iterate over the DataLoader to process data in batches
    for batch in dataloader:
        print(batch)
        # Perform your operations on each batch here, e.g., training your model
else:
    print("Data loading failed.")
