import torch
import torch.nn as nn
import torch.nn.functional as F



class VAE(nn.Module):
    def __init__(self, latent_dim=3*32*32, device='cuda'):
        super(VAE, self).__init__()
        
        self.device = device

        self.latent_dim = latent_dim
        
        # encoder layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        self.conv5 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)
        self.conv6 = nn.Conv2d(512, 1024, kernel_size=5, stride=2, padding=1)
        
        self.fc1 = nn.Linear(self.latent_dim * 3, self.latent_dim)
        self.fc2 = nn.Linear(self.latent_dim * 3, self.latent_dim)
        
        # decoder layers
        self.fc3 = nn.Linear(self.latent_dim, self.latent_dim * 3)
        self.conv7 = nn.ConvTranspose2d(1024, 1024, kernel_size=4, stride=1, padding=1)
        self.conv8 = nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1)
        self.conv9 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
        self.conv10 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.conv11 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.conv12 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.conv13 = nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1)
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def encode(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.relu(self.conv5(x))
        x = self.relu(self.conv6(x))
        x = x.view(-1, 1024 * 3 * 3)
        mean = self.fc1(x)
        logvar = self.fc2(x)
        return mean, logvar
        
    def decode(self, z):
        z = self.relu(self.fc3(z))
        z = z.view(-1, 1024, 3, 3)
        z = self.relu(self.conv7(z))
        z = self.relu(self.conv8(z))
        z = self.relu(self.conv9(z))   
        z = self.relu(self.conv10(z))
        z = self.relu(self.conv11(z))
        z = self.relu(self.conv12(z))
        z = self.sigmoid(self.conv13(z))
        return z
    
    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + std * eps

    def forward(self, x):
        # Encoder
        mu, std = self.encode(x)
        z = self.reparameterize(mu, std)
        
        # Decoder
        x_recon = self.decode(z)
        return x_recon, mu, std

    def loss_function(self, x, x_hat, mean, logvar):
        # reconstruction loss
        recon_loss = F.binary_cross_entropy(x_hat, x, reduction='sum')
        
        # KL divergence
        kl_div = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
        
        # total loss
        loss = recon_loss + kl_div
        
        return loss

    def sample(self, num_samples):
        z = torch.randn(num_samples, self.latent_dim).to(self.device)
        samples = self.decode(z)
        return samples