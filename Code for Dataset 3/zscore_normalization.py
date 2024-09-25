import pandas as pd
from sklearn.preprocessing import StandardScaler


def apply_zscore_normalization(file_path, output_file):
    """
    Apply Z-score normalization to the numeric columns of the dataset and save the result.

    Args:
        file_path (str): Path to the CSV file to be normalized.
        output_file (str): Path to save the normalized dataset.
    """
    # Load the dataset
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: The file at {file_path} was not found.")
        return
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return

    # Select only numeric columns for normalization
    numeric_columns = df.select_dtypes(include='number').columns

    # Initialize the scaler
    scaler = StandardScaler()

    # Apply Z-score normalization only to numeric columns
    df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

    # Save the standardized dataset
    df.to_csv(output_file, index=False)

    print(f"Z-score normalization applied successfully and saved to '{output_file}'")


# Define the file path for Dataset 3 (adjust accordingly)
file_path = r'C:\Users\zawlw\Downloads\FYP (EEG Data Augmentation)\Dataset 3\EEG_data.csv'
output_file = 'standardized_EEG_data.csv'

# Apply Z-score normalization to Dataset 3 and save it
apply_zscore_normalization(file_path, output_file)
