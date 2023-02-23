import torch
import matplotlib.pyplot as plt
from src.models.UNet import UNet
from src.models.DDPM import Diffusion

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


model = UNet().to('cuda')
model.load_state_dict(torch.load('results/models/model_480.pt'))





images = Diffusion(img_size=48).sample(model, 16)


# plot the 16 images in a 4x4 grid

plt.figure(figsize=(32, 32))
plt.imshow(torch.cat([
    torch.cat([i for i in images.cpu()], dim=-1),
], dim=-2).permute(1, 2, 0).cpu())
plt.show()