import torch
import torch.optim as optim
from models import Generator, Discriminator
from data_preprocessing import load_and_preprocess_data

# Set the dimensions
latent_dim = 100  # Dimension of the latent vector (input to Generator)
output_dim = 989  # Dimension of your EEG data
epochs = 100  # Number of training epochs
batch_size = 64  # Batch size
learning_rate = 0.0002  # Learning rate

# Initialize models
generator = Generator(latent_dim, output_dim)
discriminator = Discriminator(output_dim)

# Initialize optimizers
gen_optimizer = optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
disc_optimizer = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))

# Loss function
criterion = torch.nn.BCELoss()

# Load data
file_path = r'C:\Users\zawlw\Downloads\FYP (EEG Data Augmentation)\Dataset 1\standardized_mental_state.csv'  # Replace with the correct path
dataloader = load_and_preprocess_data(file_path)

# Training loop
for epoch in range(epochs):
    for i, batch in enumerate(dataloader):
        # Get real EEG data
        real_data = batch[0]  # Extract data tensor from the batch
        batch_size = real_data.size(0)

        # Create labels for real and fake data
        real_labels = torch.ones(batch_size, 1)
        fake_labels = torch.zeros(batch_size, 1)

        # Train the Discriminator
        disc_optimizer.zero_grad()

        # Discriminator loss on real data
        real_output = discriminator(real_data)
        real_loss = criterion(real_output, real_labels)

        # Generate fake data and compute discriminator loss on fake data
        latent_vector = torch.randn(batch_size, latent_dim)
        fake_data = generator(latent_vector)
        fake_output = discriminator(fake_data.detach())  # Detach to avoid training the generator on this step
        fake_loss = criterion(fake_output, fake_labels)

        # Total discriminator loss
        disc_loss = real_loss + fake_loss
        disc_loss.backward()
        disc_optimizer.step()

        # Train the Generator
        gen_optimizer.zero_grad()

        # Generate fake data and compute generator loss
        fake_output = discriminator(fake_data)
        gen_loss = criterion(fake_output,
                             real_labels)  # The generator wants the discriminator to believe the fake data is real
        gen_loss.backward()
        gen_optimizer.step()

        # Print progress
        if i % 100 == 0:
            print(
                f"Epoch [{epoch + 1}/{epochs}], Batch [{i + 1}/{len(dataloader)}], D Loss: {disc_loss.item():.4f}, G Loss: {gen_loss.item():.4f}")

torch.save(generator.state_dict(), 'generator.pth')
torch.save(discriminator.state_dict(), 'discriminator.pth')
