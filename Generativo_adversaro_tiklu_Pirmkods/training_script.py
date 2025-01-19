import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from gan_model import Generator, Discriminator

# Set random seed for reproducibility
torch.manual_seed(42)

# Device configuration (GPU if available, else CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
latent_dim = 100
batch_size = 128
lr = 0.0002
epochs = 10
num_classes = 10  # Number of classes in CIFAR-10 dataset

# Initialize generator and discriminator
generator = Generator(latent_dim + num_classes).to(device)  # Add additional input for class information
discriminator = Discriminator().to(device)

# Loss function
criterion = nn.BCELoss()

# Optimizers
optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

# Data preprocessing and loading
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = datasets.CIFAR10(root='data', train=True, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Training loop
for epoch in range(epochs):
    for i, (real_images, labels) in enumerate(train_loader):
        real_images = real_images.to(device)
        labels = labels.to(device)

        # Generate fake images
        z = torch.randn(batch_size, latent_dim, 1, 1).to(device)
        fake_labels = torch.randint(0, num_classes, (batch_size,)).to(device)  # Generate random class labels
        fake_labels_one_hot = nn.functional.one_hot(fake_labels, num_classes).float().to(device)  # Convert to one-hot encoding
        z = torch.cat([z, fake_labels_one_hot.unsqueeze(-1).unsqueeze(-1)], dim=1)  # Concatenate random noise with one-hot encoded class labels
        fake_images = generator(z)


        # Train discriminator
        optimizer_D.zero_grad()
        real_labels = torch.ones(real_images.size(0), 1, 1, 1).to(device)  # Adjust batch size dynamically

        # Ensure real_labels has the same size as real_output
        real_labels = real_labels.view(-1, 1, 1, 1)

        fake_labels = torch.zeros(fake_images.size(0), 1, 1, 1).to(device)  # Adjust batch size dynamically

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

        # Ensure real_labels has the same size as output
        real_labels = torch.ones(fake_images.size(0), 1, 1, 1).to(device)

        # Calculate generator loss
        g_loss = criterion(output, real_labels)

        g_loss.backward()
        optimizer_G.step()
        
        # Print training progress
        if (i + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train_loader)}], '
                  f'D_loss: {d_loss.item():.4f}, G_loss: {g_loss.item():.4f}')

    # Save generated images at the end of each epoch
    save_image(fake_images[:25], f'generated_images_epoch_{epoch+1}.png', nrow=5, normalize=True)
    torch.save(generator.state_dict(), 'generator.pth')  # Save the generator's state dictionary
