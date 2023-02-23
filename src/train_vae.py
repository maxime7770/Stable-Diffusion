import os
import torch
from tqdm import tqdm
from utils import get_data_vae
from models.VAE import VAE





def train(img_size, batch_size, num_workers, dataset_path, train_folder, val_folder, slice_size, epochs, lr, save_path, device):
    data_loader, _ = get_data_vae(img_size, batch_size, num_workers, dataset_path, train_folder, val_folder, slice_size)
    model = VAE(device=device).to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    for epoch in range(epochs):
        total_loss = 0
        for i, images in enumerate(tqdm(data_loader)):
            optimizer.zero_grad()
            images = images.to(device)
            reconstructed_images, mean, logvar = model(images)
            loss = model.loss_function(images, reconstructed_images, mean, logvar)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print("Epoch [{}/{}], Loss: {:.4f}".format(epoch+1, epochs, total_loss/len(data_loader.dataset)))

    torch.save(model.state_dict(), os.path.join(save_path+'/models', f"model_vae.pt"))



if __name__ == '__main__':
    img_size = 256
    batch_size = 8
    num_workers = 2
    dataset_path = 'datasets/landscape'
    train_folder = 'landscape'
    val_folder = 'val'
    slice_size = 1
    epochs = 100
    lr = 1e-4
    save_path = 'results'
    device = 'cuda'
    train(img_size, batch_size, num_workers, dataset_path, train_folder, val_folder, slice_size, epochs, lr, save_path, device)
