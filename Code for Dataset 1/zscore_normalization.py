import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load the dataset with the corrected file path
df = pd.read_csv(r'C:\Users\zawlw\Downloads\FYP (EEG Data Augmentation)\Dataset 1\mental-state.csv')

# Initialize the scaler
scaler = StandardScaler()

# Apply Z-score normalization to each channel
df_standardized = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

# Save the standardized dataset if needed
df_standardized.to_csv('standardized_mental_state.csv', index=False)
