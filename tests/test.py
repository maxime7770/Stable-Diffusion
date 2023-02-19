import torch
from torch import nn


input = torch.randn(1, 3, 2, 2)
m = nn.GroupNorm(3, 3)


print(input)
print(m(input))