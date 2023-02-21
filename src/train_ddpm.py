import os
import torch
import torch.nn as nn
from tqdm import tqdm
from models.UNet import UNet
from models.DDPM import Diffusion
from utils import get_data, save_images



def train(img_size, batch_size, num_workers, dataset_path, train_folder, val_folder, slice_size, epochs, lr, save_path, device):
    data_loader, _ = get_data(img_size, batch_size, num_workers, dataset_path, train_folder, val_folder, slice_size)
    model = UNet(device=device).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    diffusion = Diffusion(img_size=img_size, device=device)
    
    for epoch in range(epochs):
        print(f'==== EPOCH {epoch} ====')
        for i, images in enumerate(tqdm(data_loader)):
            images = images.to(device)
            time_steps = diffusion.sample_time_steps(images.shape[0]).to(device)
            x_t, noise = diffusion.noise_images(images, time_steps)
            predicted_noise = model(x_t, time_steps)
            loss = criterion(predicted_noise, noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch % 10 == 0:
            print('Sample images')
            sampled_images = diffusion.sample(model, images.shape[0])
            print('Save images')
            save_images(sampled_images, os.path.join(save_path+'/images', f"epoch_{epoch}.png"))
            print('Save model')
            torch.save(model.state_dict(), os.path.join(save_path+'/models', f"model_{epoch}.pt"))



if __name__ == '__main__':
    os.makedirs('./results/images', exist_ok=True)
    os.makedirs('./results/models', exist_ok=True)
    img_size = 32
    batch_size = 8
    num_workers = 1
    dataset_path = 'datasets/landscape'
    train_folder = 'train'
    val_folder = 'val'
    slice_size = 1
    epochs = 500
    lr = 1e-4
    save_path = 'results'
    device = 'cuda'
    train(img_size, batch_size, num_workers, dataset_path, train_folder, val_folder, slice_size, epochs, lr, save_path, device)
