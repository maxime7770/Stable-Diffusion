import os
import glob
import shutil
import random
import torchvision.datasets as datasets



# load celebA dataset   

celebA = datasets.CelebA('./datasets/', split='all', download=True)


def split_dataset(folder, path_train, path_val, path_test, split=(0.8, 0.1, 0.1)):

    # create train, val, test folders
    os.makedirs(path_train, exist_ok=True)
    os.makedirs(path_val, exist_ok=True)
    os.makedirs(path_test, exist_ok=True)

    # get all image paths
    image_paths = glob.glob(os.path.join(folder, '*.jpg'))

    # shuffle image paths
    random.shuffle(image_paths)

    # split image paths
    split_1 = int(split[0] * len(image_paths))
    split_2 = int((split[0] + split[1]) * len(image_paths))
    image_paths_train = image_paths[:split_1]
    image_paths_val = image_paths[split_1:split_2]
    image_paths_test = image_paths[split_2:]

    # copy images to train, val, test folders
    for image_path in image_paths_train:
        shutil.copy(image_path, path_train)
    for image_path in image_paths_val:
        shutil.copy(image_path, path_val)
    for image_path in image_paths_test:
        shutil.copy(image_path, path_test)

    


if __name__ == '__main__':
    split_dataset('./datasets/celeba/img_align_celeba', './datasets/celeba/train', './datasets/celeba/val', './datasets/celeba/test')