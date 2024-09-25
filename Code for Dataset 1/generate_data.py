import torch
import pandas as pd  # Import pandas
from models import Generator

# Set the dimensions
latent_dim = 100  # The dimension of the latent vector (must match training configuration)
output_dim = 989  # The output dimension should match your EEG data's dimensionality

# Initialize the Generator model
generator = Generator(latent_dim, output_dim)

# Load the trained model's state
try:
    generator.load_state_dict(torch.load('generator.pth', weights_only=True))
except FileNotFoundError:
    print("Error: The file 'generator.pth' was not found. Make sure you have trained and saved the model properly.")
    exit(1)

# Set the generator to evaluation mode
generator.eval()

# Function to generate new data
def generate_new_data(generator, num_samples=10):
    latent_vectors = torch.randn(num_samples, latent_dim)
    with torch.no_grad():
        generated_data = generator(latent_vectors)
    return generated_data

# Generate 10 new samples
new_data = generate_new_data(generator, num_samples=10)

# Print or save the generated data
print(new_data)

# Convert tensor to DataFrame and save to a CSV file
generated_data_df = pd.DataFrame(new_data.numpy())
generated_data_df.to_csv('generated_eeg_data.csv', index=False)
