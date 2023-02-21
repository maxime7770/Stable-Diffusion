import os
import glob
import shutil
import random
import torchvision.datasets as datasets
from pathlib import Path
from PIL import Image


def get_kaggle_dataset(dataset_path, # Local path to download dataset to
                dataset_slug, # Dataset slug (ie "zillow/zecon")
                unzip=True, # Should it unzip after downloading?
                force=False # Should it overwrite or error if dataset_path exists?
               ):
    '''Downloads an existing dataset and metadata from kaggle'''
    from kaggle import api
    if not force and os.path.exists(dataset_path):
        return dataset_path
    api.dataset_metadata(dataset_slug, str(dataset_path))
    api.dataset_download_files(dataset_slug, str(dataset_path))
    if unzip:
        zipped_file = Path(dataset_path)/f"{dataset_slug.split('/')[-1]}.zip"
        import zipfile
        with zipfile.ZipFile(zipped_file, 'r') as zip_ref:
            zip_ref.extractall(Path(dataset_path+'/landscape'))
        zipped_file.unlink()


def cifar10_loader():
    cifar10 = datasets.CIFAR10('./datasets/', download=True)
    return cifar10

def stanfordcars_loader():
    stanfordcars = datasets.StanfordCars('./datasets/', download=True)
    return stanfordcars

def celeba_loader():
    celebA = datasets.CelebA('./datasets/', split='all', download=True)
    return celebA

def landscape_loader():
    get_kaggle_dataset('./datasets/landscape', 'arnaud58/landscape-pictures', unzip=True, force=False)
    return 



def split_dataset(folder, path_train, path_val, path_test, split=(0.8, 0.1, 0.1)):

    # create train, val, test folders
    os.makedirs(path_train, exist_ok=True)
    os.makedirs(path_val, exist_ok=True)
    os.makedirs(path_test, exist_ok=True)

    # get all image paths
    image_paths = glob.glob(os.path.join(folder, '*.jpg'))

    # transform grey images to rgb
    for image_path in image_paths:
        image = Image.open(image_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')
            image.save(image_path)

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
    landscape_loader()
    split_dataset('./datasets/landscape/landscape', './datasets/landscape/train', './datasets/landscape/val', './datasets/landscape/test')