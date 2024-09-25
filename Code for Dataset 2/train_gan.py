import torch
import torch.optim as optim
import torch.nn as nn
from models import Generator, Discriminator
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Hyperparameters
batch_size = 64
z_dim = 100  # Dimension of the noise vector
learning_rate = 0.0002
beta1 = 0.5  # For Adam optimizer

# DataLoader
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Assuming you are using MNIST as a placeholder
dataset = datasets.MNIST(root='./data', download=True, train=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize models
generator = Generator(input_dim=z_dim, output_dim=784)  # Adjust output_dim as per your dataset
discriminator = Discriminator(input_dim=784)  # Adjust input_dim as per your dataset

# Initialize optimizers
optimizer_g = optim.Adam(generator.parameters(), lr=learning_rate, betas=(beta1, 0.999))
optimizer_d = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(beta1, 0.999))

# Loss function
criterion = nn.BCELoss()

# Training loop
num_epochs = 50  # Adjust based on your needs
for epoch in range(num_epochs):
    for i, (imgs, _) in enumerate(dataloader):
        # Get batch size
        batch_size = imgs.size(0)

        # Create labels
        real_labels = torch.ones(batch_size, 1)
        fake_labels = torch.zeros(batch_size, 1)

        # Reshape images
        imgs = imgs.view(batch_size, -1)  # Flatten images

        # Train Discriminator
        optimizer_d.zero_grad()

        outputs = discriminator(imgs)
        d_loss_real = criterion(outputs, real_labels)
        d_loss_real.backward()

        z = torch.randn(batch_size, z_dim)
        fake_imgs = generator(z)
        outputs = discriminator(fake_imgs.detach())
        d_loss_fake = criterion(outputs, fake_labels)
        d_loss_fake.backward()
        optimizer_d.step()

        # Train Generator
        optimizer_g.zero_grad()

        outputs = discriminator(fake_imgs)
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
