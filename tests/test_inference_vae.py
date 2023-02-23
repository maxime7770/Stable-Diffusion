import torch
import matplotlib.pyplot as plt
from src.models.VAE import VAE
from PIL import Image
import torchvision.transforms as transforms

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


model = VAE().to('cuda')
model.load_state_dict(torch.load('results/models/model_vae.pt'))





image = Image.open('datasets/landscape/landscape/00000000_(3).jpg')
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((256, 256)),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
  
image = transform(image).to('cuda')

print(image)

model.eval()
with torch.no_grad():
    reconstruction = model(image)[0][0]

print(reconstruction.shape)

# plot the image and the reconstruction side by side

plt.figure(figsize=(32, 32))
plt.imshow(torch.cat([
    torch.cat([image, reconstruction], dim=-1),
], dim=-2).permute(1, 2, 0).cpu())
plt.show()