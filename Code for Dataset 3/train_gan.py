import torch
import torch.optim as optim
import torch.nn as nn
from models import Generator, Discriminator
from torch.utils.data import DataLoader
from torchvision import transforms
import pandas as pd

# Hyperparameters
batch_size = 64
z_dim = 100  # Dimension of the noise vector
learning_rate = 0.0002
beta1 = 0.5  # For Adam optimizer
output_dim = 15  # Number of features in your EEG dataset
num_epochs = 100  # Number of epochs to train the GAN, define it here

# Initialize models
generator = Generator(input_dim=z_dim, output_dim=output_dim)
discriminator = Discriminator(input_dim=output_dim)


# Load EEG Dataset 3
def load_eeg_data(file_path):
    df = pd.read_csv(file_path)
    df = df.apply(pd.to_numeric, errors='coerce')
    df.fillna(df.mean(), inplace=True)  # Handle missing values
    data = torch.tensor(df.values, dtype=torch.float32)
    return data

# Assuming 'standardized_EEG_data.csv' is the normalized EEG dataset
file_path = 'standardized_EEG_data.csv'
eeg_data = load_eeg_data(file_path)

# Create DataLoader
dataloader = DataLoader(eeg_data, batch_size=batch_size, shuffle=True)

# Initialize models
generator = Generator(input_dim=z_dim, output_dim=output_dim)
discriminator = Discriminator(input_dim=output_dim)

# Initialize optimizers
optimizer_g = optim.Adam(generator.parameters(), lr=learning_rate, betas=(beta1, 0.999))
optimizer_d = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(beta1, 0.999))

# Loss function
criterion = nn.BCELoss()

# Training loop
for epoch in range(num_epochs):
    for i, batch in enumerate(dataloader):
        # Get batch size
        batch_size = batch.size(0)

        # Real and fake labels
        real_labels = torch.ones(batch_size, 1)
        fake_labels = torch.zeros(batch_size, 1)

        # Train Discriminator
        optimizer_d.zero_grad()

        outputs = discriminator(batch)
        d_loss_real = criterion(outputs, real_labels)
        d_loss_real.backward()

        # Generate fake data
        z = torch.randn(batch_size, z_dim)
        fake_data = generator(z)
        outputs = discriminator(fake_data.detach())
        d_loss_fake = criterion(outputs, fake_labels)
        d_loss_fake.backward()
        optimizer_d.step()

        # Train Generator
        optimizer_g.zero_grad()

        outputs = discriminator(fake_data)
        g_loss = criterion(outputs, real_labels)
        g_loss.backward()
        optimizer_g.step()

        # Print progress
        if (i + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(dataloader)}], '
                  f'D Loss: {d_loss_real.item() + d_loss_fake.item()}, G Loss: {g_loss.item()}')

# Save the models
torch.save(generator.state_dict(), 'generator.pth')
torch.save(discriminator.state_dict(), 'discriminator.pth')
