import os.path
import torch
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
import numpy as np
import torchvision.transforms as T
import random
import matplotlib.pyplot as plt
from PIL import Image
import cv2 as cv

np.set_printoptions(threshold=np.inf)


class Data_loader(Dataset):
    def __init__(self, root, train=True, test=False):
        train_paths = list(np.load(os.path.join(root, 'train_path.npy')))
        test_paths = list(np.load(os.path.join(root, 'test_path.npy')))
        val_paths = list(np.load(os.path.join(root, 'val_path.npy')))
        self.root = root

        self.flag = False
        if test:
            self.image_paths = test_paths
        elif train:
            self.flag = True
            self.image_paths = train_paths
        else:
            self.image_paths = val_paths

        print(self.root, ' loading data')
        self.images, self.labels = self.load_data(self.image_paths)

    def load_data(self, data_path):
        images = []
        labels = []
        for i in data_path:
            image_path = i
            image_path = image_path.replace('./', self.root)
            image_path = image_path.replace('\\', '/')
            label_path = image_path.replace('image', 'mask')
            image = Image.open(image_path).convert('L')
            image = self.clahe_gamma(image)
            image = np.array(image, 'f')
            label = Image.open(label_path).convert('L')
            label = np.array(label, 'f')
            label[label <= 128] = 0
            label[label > 128] = 1
            if not np.all(label==0):
                images.append(image)
                labels.append(label)

        return images, labels

    def augment(self, input, flipCode, rotate, flag=0):
        rows, cols = input.shape
        if flag == 0:
            flags = cv.INTER_LINEAR
        else:
            flags = cv.INTER_NEAREST
        if flipCode == 0:
            flip_out = np.fliplr(input)
        elif flipCode == 1:
            flip_out = cv.warpAffine(input, rotate, (cols, rows), flags=flags)
        elif flipCode == 2:
            flip_out = np.fliplr(input)
            flip_out = cv.warpAffine(flip_out, rotate, (cols, rows), flags=flags)
        return flip_out

    def clahe_gamma(self, img, gamma=1.0):
        clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        imgs_equalized = clahe.apply(np.array(img, dtype=np.uint8))
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        new_imgs = cv.LUT(np.array(imgs_equalized, dtype=np.uint8), table)
        return new_imgs

    def _downSample(self, image):
        image = torch.tensor(image, dtype=torch.float32)
        image = torch.unsqueeze(image, dim=0)
        mp = torch.nn.MaxPool2d(2)
        image = mp(image)
        image = torch.squeeze(image, dim=0)
        return np.array(image)

    def __getitem__(self, item):
        image, label = self.images[item], self.labels[item]

        image = self._downSample(image)
        label = self._downSample(label)

        # if self.flag:
        #     flipCode = random.choice([-1, 0, 1, 2])
        # else:
        #     flipCode = -1
        # if flipCode != -1:
        #     rows, cols = image.shape
        #     t = random.randrange(0, 180, 15)
        #     rotate = cv.getRotationMatrix2D((rows * 0.5, cols * 0.5), t, 1)
        #     image = self.augment(image, flipCode, rotate, 0)
        #     label = self.augment(label.astype(np.float64), flipCode, rotate, 1)
        max_image = image.max()
        min_image = image.min()
        image = (image - min_image) / (max_image - min_image)
        norm_transform = T.Compose([
            T.ToTensor(),
            T.Normalize((0.5,), (0.5,))
        ])
        transform = T.ToTensor()
        image = norm_transform(image.copy())
        label = transform(label.copy())
        # print(np.array(label))

        return image, label

    def __len__(self):
        return len(self.images)

class Data_loader_joint(Dataset):
    def __init__(self, root, datasets, train=True, test=False):
        self.root = root
        # datasets = {0: 'Chase', 1: 'Stare', 2: 'Rite', 3: 'Drhagis'}
        all_train_paths = []
        all_test_paths = []
        all_val_paths = []

        for i in datasets:
            train_paths = list(np.load(root + f'{datasets[i]}/train_path.npy'))
            test_paths = list(np.load(root + f'{datasets[i]}/test_path.npy'))
            val_paths = list(np.load(root + f'{datasets[i]}/val_path.npy'))
            
            train_paths = [(datasets[i], k) for k in train_paths]
            test_paths = [(datasets[i], k) for k in test_paths]
            val_paths = [(datasets[i], k) for k in val_paths]

            all_train_paths.extend(train_paths)
            all_test_paths.extend(test_paths)
            all_val_paths.extend(val_paths)

        self.flag = False
        if test:
            self.image_paths = all_test_paths
        elif train:
            self.flag = True
            self.image_paths = all_train_paths
        else:
            self.image_paths = all_val_paths

        print(self.root, ' loading data')
        self.images, self.labels = self.load_data(self.image_paths)

    def load_data(self, data_path):
        images = []
        labels = []
        for type, i in data_path:
            image_path = i
            image_path = image_path.replace('./', self.root + f'{type}/')
            image_path = image_path.replace('\\', '/')
            label_path = image_path.replace('image', 'mask')
            image = Image.open(image_path).convert('L')
            image = self.clahe_gamma(image)
            image = np.array(image, 'f')
            label = Image.open(label_path).convert('L')
            label = np.array(label, 'f')
            label[label <= 128] = 0
            label[label > 128] = 1
            if not np.all(label==0):
                images.append(image)
                labels.append(label)

        return images, labels

    def augment(self, input, flipCode, rotate, flag=0):
        rows, cols = input.shape
        if flag == 0:
            flags = cv.INTER_LINEAR
        else:
            flags = cv.INTER_NEAREST
        if flipCode == 0:
            flip_out = np.fliplr(input)
        elif flipCode == 1:
            flip_out = cv.warpAffine(input, rotate, (cols, rows), flags=flags)
        elif flipCode == 2:
            flip_out = np.fliplr(input)
            flip_out = cv.warpAffine(flip_out, rotate, (cols, rows), flags=flags)
        return flip_out

    def clahe_gamma(self, img, gamma=1.0):
        clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        imgs_equalized = clahe.apply(np.array(img, dtype=np.uint8))
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        new_imgs = cv.LUT(np.array(imgs_equalized, dtype=np.uint8), table)
        return new_imgs

    def _downSample(self, image):
        image = torch.tensor(image, dtype=torch.float32)
        image = torch.unsqueeze(image, dim=0)
        mp = torch.nn.MaxPool2d(2)
        image = mp(image)
        image = torch.squeeze(image, dim=0)
        return np.array(image)

    def __getitem__(self, item):
        image, label = self.images[item], self.labels[item]

        image = self._downSample(image)
        label = self._downSample(label)

        # if self.flag:
        #     flipCode = random.choice([-1, 0, 1, 2])
        # else:
        #     flipCode = -1
        # if flipCode != -1:
        #     rows, cols = image.shape
        #     t = random.randrange(0, 180, 15)
        #     rotate = cv.getRotationMatrix2D((rows * 0.5, cols * 0.5), t, 1)
        #     image = self.augment(image, flipCode, rotate, 0)
        #     label = self.augment(label.astype(np.float64), flipCode, rotate, 1)
        max_image = image.max()
        min_image = image.min()
        image = (image - min_image) / (max_image - min_image)
        norm_transform = T.Compose([
            T.ToTensor(),
            T.Normalize((0.5,), (0.5,))
        ])
        transform = T.ToTensor()
        image = norm_transform(image.copy())
        label = transform(label.copy())
        # print(np.array(label))

        return image, label

    def __len__(self):
        return len(self.images)