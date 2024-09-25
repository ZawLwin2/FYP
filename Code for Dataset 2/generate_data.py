import torch
import numpy as np
from models import Generator

# Parameters
latent_dim = 100  # Dimension of the latent vector (input to Generator)
output_dim = 784  # Dimension of the data as per the saved model
num_samples = 1000  # Number of samples to generate (adjust as needed)
model_path = 'generator.pth'  # Path to the saved Generator model

# Load the trained Generator model
generator = Generator(input_dim=latent_dim, output_dim=output_dim)
generator.load_state_dict(torch.load(model_path))
generator.eval()  # Set the model to evaluation mode

# Generate new data
with torch.no_grad():
    # Create random latent vectors
    latent_vectors = torch.randn(num_samples, latent_dim)

    # Generate data
    generated_data = generator(latent_vectors)

    # Convert the generated data to a numpy array
    generated_data_np = generated_data.numpy()

# Optionally, save the generated data to a CSV file
np.savetxt('generated_data.csv', generated_data_np, delimiter=',')

print(f"Generated {num_samples} samples and saved to 'generated_data.csv'")
