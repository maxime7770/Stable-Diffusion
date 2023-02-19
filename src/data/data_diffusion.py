import torchvision.datasets as datasets



# load celebA dataset   

celebA = datasets.CelebA('./datasets/', split='all', download=True)