import torch
import torch.nn as nn

# Define the Generator model
class Generator(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim * 2, hidden_dim * 4),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim * 4, output_dim),
            nn.Tanh()  # Tanh to scale output to [-1, 1]
        )

    def forward(self, x):
        return self.model(x)

# Define the Discriminator model
class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # Sigmoid to output probability between [0, 1]
        )

    def forward(self, x):
        return self.model(x)

# Initialize models
input_dim = 100  # Example latent vector dimension
output_dim = 989  # Dimension of EEG data

generator = Generator(input_dim, output_dim)
discriminator = Discriminator(output_dim)

# Initialize optimizers
gen_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
disc_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

# Define the loss function
criterion = nn.BCELoss()
