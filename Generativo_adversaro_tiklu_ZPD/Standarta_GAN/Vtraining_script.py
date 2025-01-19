import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from Vgan_model import Generator, Discriminator
import numpy as np
from collections import defaultdict

# Set random seed for reproducibility
torch.manual_seed(42)

# Device configuration (GPU if available, else CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
latent_dim = 100
batch_size = 128
lr_gen = 0.00005
lr_disc = 0.0002
epochs = 200
image_dim = 32 * 32 * 3

# Initialize generator and discriminator
generator = Generator(latent_dim, image_dim).to(device)  # Add additional input for class information
discriminator = Discriminator(image_dim).to(device)

# Loss function
criterion = nn.BCELoss()

# Optimizers
optimizer_G = optim.Adam(generator.parameters(), lr=lr_gen, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr_disc, betas=(0.5, 0.999))

# Data preprocessing and loading
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

svhn_dataset = datasets.SVHN(root='data', split='train', transform=transform, download=True)

def limit_dataset(dataset, num_per_class):
    class_counts = defaultdict(int)
    limited_indices = []
    for idx, (_, label) in enumerate(dataset):
        label = int(label)
        if class_counts[label] < num_per_class:
            limited_indices.append(idx)
            class_counts[label] += 1
    return torch.utils.data.Subset(dataset, limited_indices)

train_dataset = limit_dataset(svhn_dataset, num_per_class=1000)

# DataLoader
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Training loop
for epoch in range(epochs):
    for i, (real_images, _) in enumerate(train_loader):
        real_images = real_images.view(real_images.size(0), -1).to(device)
        
        # Generate fake images
        z = torch.randn(batch_size, latent_dim).to(device)
        fake_images = generator(z)
        
        
        # Train discriminator
        optimizer_D.zero_grad()
        real_labels = torch.ones(real_images.size(0), 1).to(device)
        fake_labels = torch.zeros(fake_images.size(0), 1).to(device)


        real_output = discriminator(real_images)
        fake_output = discriminator(fake_images.detach())

        real_loss = criterion(real_output, real_labels)
        fake_loss = criterion(fake_output, fake_labels)
        d_loss = real_loss + fake_loss
        d_loss.backward()
        optimizer_D.step()

        # Train generator
        optimizer_G.zero_grad()
        output = discriminator(fake_images)

        # Calculate generator loss
        g_loss = criterion(output, torch.ones(fake_images.size(0), 1).to(device))

        g_loss.backward()
        optimizer_G.step()
        
        # Print training progress
        print_interval = max(len(train_loader) // 10, 1)
        if (i + 1) % print_interval == 0 or (i + 1) == len(train_loader):
            print(f'Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train_loader)}], '
                  f'D_loss: {d_loss.item():.4f}, G_loss: {g_loss.item():.4f}')

    # Save generated images at the end of each epoch
    if (epoch + 1) % 10 ==0:
        save_image(fake_images.view(fake_images.size(0), 3, 32, 32)[:25], f'path/generated_images_epoch_{epoch+1}.png', nrow=5, normalize=True)
        torch.save(generator.state_dict(), f'path/generator_{epoch+1}.pth')
        torch.save(discriminator.state_dict(), f'path/discriminator_{epoch+1}.pth')
        torch.save(optimizer_G.state_dict(), f'path/optimizer_G_{epoch+1}.pth')
        torch.save(optimizer_D.state_dict(), f'path/optimizer_D_{epoch+1}.pth')
