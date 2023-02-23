import os
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
from PIL import Image



class CustomDataset(Dataset):

    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.images = os.listdir(root)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = os.path.join(self.root, self.images[idx])
        image = Image.open(image_path)
        if self.transform:
            image = self.transform(image)
        return image



def plot_images(images):
    plt.figure(figsize=(32, 32))
    plt.imshow(torch.cat([
        torch.cat([i for i in images.cpu()], dim=-1),
    ], dim=-2).permute(1, 2, 0).cpu())
    plt.show()


def save_images(images, path, **kwargs):
    grid = torchvision.utils.make_grid(images, **kwargs)
    ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
    im = Image.fromarray(ndarr)
    im.save(path)


def get_data(img_size, batch_size, num_workers, dataset_path, train_folder, val_folder, slice_size):
    train_transforms = torchvision.transforms.Compose([
        T.Resize(img_size + int(.25*img_size)), 
        T.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    val_transforms = torchvision.transforms.Compose([
        T.Resize(img_size),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    train_dataset = CustomDataset(os.path.join(dataset_path, train_folder), transform=train_transforms)
    val_dataset = CustomDataset(os.path.join(dataset_path, val_folder), transform=val_transforms)
    
    if slice_size > 1:
        train_dataset = torch.utils.data.Subset(train_dataset, indices=range(0, len(train_dataset), slice_size))
        val_dataset = torch.utils.data.Subset(val_dataset, indices=range(0, len(val_dataset), slice_size))

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_dataset = DataLoader(val_dataset, batch_size=2*batch_size, shuffle=False, num_workers=num_workers)
    return train_dataloader, val_dataset



def get_data_vae(img_size, batch_size, num_workers, dataset_path, train_folder, val_folder, slice_size):
    train_transforms = torchvision.transforms.Compose([
        T.Resize((img_size, img_size)), 
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    val_transforms = torchvision.transforms.Compose([
        T.Resize((img_size,  img_size)),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    train_dataset = CustomDataset(os.path.join(dataset_path, train_folder), transform=train_transforms)
    val_dataset = CustomDataset(os.path.join(dataset_path, val_folder), transform=val_transforms)
    
    if slice_size > 1:
        train_dataset = torch.utils.data.Subset(train_dataset, indices=range(0, len(train_dataset), slice_size))
        val_dataset = torch.utils.data.Subset(val_dataset, indices=range(0, len(val_dataset), slice_size))

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_dataset = DataLoader(val_dataset, batch_size=2*batch_size, shuffle=False, num_workers=num_workers)
    return train_dataloader, val_dataset