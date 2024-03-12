import os
import shutil
import random
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2
from torchvision.transforms import transforms
import torchvision.transforms as T
import torchvision.transforms.functional as TF


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['PYTHONHASHSEED'] = str(seed)


class Data_patch(Dataset):
    def __init__(self, root):
        self.root = root
        self.image_paths = os.listdir(os.path.join(root, 'image'))

    def transform(self, image, mask):
        # Random crop
        i, j, h, w = transforms.RandomCrop.get_params(image, output_size=(256, 256))
        image = TF.crop(image, i, j, h, w)
        mask = TF.crop(mask, i, j, h, w)
        return image, mask

    def __getitem__(self, item):
        image_path = self.image_paths[item]
        image = Image.open(os.path.join(self.root, 'image', image_path))
        label = Image.open(os.path.join(self.root, 'mask', image_path))

        random_image, random_label = self.transform(image, label)

        return random_image, random_label

    def __len__(self):
        return len(self.image_paths)


if __name__ == '__main__':
    setup_seed(20)
    root = './'
    patch_num_per_image = 50
    datasets = ['Rite-ori', 'Stare-ori', 'Drhagis-ori']
    for i in datasets:
        ds = Data_patch(os.path.join(root, i))
        image_patchs_path = os.path.join(root, i, 'image_patchs')
        mask_patchs_path = os.path.join(root, i, 'mask_patchs')
        os.mkdir(image_patchs_path)
        os.mkdir(mask_patchs_path)
        for j in range(len(ds)):
            os.mkdir(os.path.join(image_patchs_path, ds.image_paths[j].replace('.jpg', '')))
            os.mkdir(os.path.join(mask_patchs_path, ds.image_paths[j].replace('.jpg', '')))
            for p in range(patch_num_per_image):
                image, label = ds[j]
                image.save(os.path.join(image_patchs_path, ds.image_paths[j].replace('.jpg', ''), f'{p}.jpg'))
                label.save(os.path.join(mask_patchs_path, ds.image_paths[j].replace('.jpg', ''), f'{p}.jpg'))



