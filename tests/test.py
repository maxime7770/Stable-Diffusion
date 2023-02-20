import torch
from torch import nn



input = torch.randn(1, 3, 2, 2)
m = nn.GroupNorm(3, 3)


print(input)
print(m(input))




input = torch.arange(1, 5, dtype=torch.float32).view(1, 1, 2, 2)

m = nn.Upsample(scale_factor=2, mode='nearest')
m2 = m = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

print(input)
print(m(input))
print(m2(input))