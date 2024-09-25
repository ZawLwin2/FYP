import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load the dataset with the corrected file path
df = pd.read_csv(r'C:\Users\zawlw\Downloads\FYP (EEG Data Augmentation)\Dataset 2\emotions.csv')

# Select only numeric columns for normalization
numeric_columns = df.select_dtypes(include='number').columns

# Initialize the scaler
scaler = StandardScaler()

# Apply Z-score normalization only to numeric columns
df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

# Save the standardized dataset if needed
df.to_csv('standardized_mental_state.csv', index=False)

print("Z-score normalization applied successfully and saved to 'standardized_mental_state.csv'")
