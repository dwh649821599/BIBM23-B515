from torch.utils.data import Dataset
import nibabel as nib
import numpy as np
import torchvision.transforms as T
import torch
import random
import matplotlib.pyplot as plt
import cv2 as cv
import os
from PIL import Image


class Data_loader(Dataset):
    def __init__(self, root, train=True, test=False):

        train_paths = list(np.load(root + 'train_path.npy'))
        test_paths = list(np.load(root + 'test_path.npy'))
        val_paths = list(np.load(root + 'val_path.npy'))
        self.root = root

        if 'Sunnybrook' in self.root:
            paths = [train_paths, test_paths, val_paths]
            retain_list = []
            for i, set in enumerate(paths):
                retain_list.append([])
                for j, image in enumerate(set):
                    if './' in image:
                        image_path = image.replace('./', self.root)
                    else:
                        image_path = image.replace('.\\', self.root)
                    image_path = image_path.replace('\\', '/')
                    label_path = image_path.replace('image', 'label')
                    label = nib.load(label_path)
                    label = label.get_fdata(caching='unchanged')
                    if np.max(label) == 2:
                        retain_list[i].append(j)
            train_paths = [train_paths[k] for k in retain_list[0]]
            test_paths = [test_paths[k] for k in retain_list[1]]
            val_paths = [val_paths[k] for k in retain_list[2]]
        
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
            if './' in image_path:
                image_path = image_path.replace('./', self.root)
            else:
                image_path = image_path.replace('.\\', self.root)
            image_path = image_path.replace('\\', '/')
            label_path = image_path.replace('image', 'label')
            image = nib.load(image_path)
            label = nib.load(label_path)
            image = image.get_fdata(caching='unchanged')
            label = label.get_fdata(caching='unchanged')
            import cv2
            label[label != 1] == 0
            label[label == 1] = 255
            cv2.imwrite('image.png', image)
            cv2.imwrite('label.png', label)
            exit()
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

    def _downSample(self, image):
        image = torch.tensor(image)
        image = torch.unsqueeze(image, dim=0)
        mp = torch.nn.MaxPool2d(2)
        image = mp(image)
        image = torch.squeeze(image, dim=0)
        return np.array(image)

    def _ACDC_downSample(self, image, label):
        img = cv.resize(image, (128, 128), interpolation=cv.INTER_LINEAR)
        lb = cv.resize(label, (128, 128), interpolation=cv.INTER_NEAREST)
        return img, lb

    def __getitem__(self, item):
        image, label = self.images[item], self.labels[item]
        if 'ACDC' in self.root:
            image, label = self._ACDC_downSample(image, label)
        else:
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
        return image, label

    def __len__(self):
        return len(self.images)


class Data_loader_joint(Dataset):
    def __init__(self, root, datasets, train=True, test=False):
        # datasets = {0: 'Sunnybrook', 1: 'CAP', 2: 'ACDC', 3: 'M&M'}
        all_train_paths = []
        all_test_paths = []
        all_val_paths = []
        self.root = root

        for idx in datasets:  
            train_paths = list(np.load(root + f'{datasets[idx]}/train_path.npy'))
            test_paths = list(np.load(root + f'{datasets[idx]}/test_path.npy'))
            val_paths = list(np.load(root + f'{datasets[idx]}/val_path.npy'))

            if datasets[idx] == 'Sunnybrook':
                paths = [train_paths, test_paths, val_paths]
                retain_list = []
                for i, set in enumerate(paths):
                    retain_list.append([])
                    for j, image in enumerate(set):
                        if './' in image:
                            image_path = image.replace('./', self.root + f'{datasets[idx]}/')
                        else:
                            image_path = image.replace('.\\', self.root + f'{datasets[idx]}/')
                        image_path = image_path.replace('\\', '/')
                        label_path = image_path.replace('image', 'label')
                        label = nib.load(label_path)
                        label = label.get_fdata(caching='unchanged')
                        if np.max(label) == 2:
                            retain_list[i].append(j)
                train_paths = [(datasets[idx], train_paths[k]) for k in retain_list[0]]
                test_paths = [(datasets[idx], test_paths[k]) for k in retain_list[1]]
                val_paths = [(datasets[idx], val_paths[k]) for k in retain_list[2]]
            else:
                train_paths = [(datasets[idx], k) for k in train_paths]
                test_paths = [(datasets[idx], k) for k in test_paths]
                val_paths = [(datasets[idx], k) for k in val_paths]

            # print(test_paths)
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
            if './' in image_path:
                image_path = image_path.replace('./', self.root + f'{type}/')
            else:
                image_path = image_path.replace('.\\', self.root + f'{type}/')
            image_path = image_path.replace('\\', '/')
            label_path = image_path.replace('image', 'label')
            image = nib.load(image_path)
            label = nib.load(label_path)
            image = image.get_fdata(caching='unchanged')
            label = label.get_fdata(caching='unchanged')
            if not np.all(label==0):
                images.append((type, image))
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

    def _downSample(self, image):
        image = torch.tensor(image)
        image = torch.unsqueeze(image, dim=0)
        mp = torch.nn.MaxPool2d(2)
        image = mp(image)
        image = torch.squeeze(image, dim=0)
        return np.array(image)

    def _ACDC_downSample(self, image, label):
        img = cv.resize(image, (128, 128), interpolation=cv.INTER_LINEAR)
        lb = cv.resize(label, (128, 128), interpolation=cv.INTER_NEAREST)
        return img, lb

    def __getitem__(self, item):
        type_image, label = self.images[item], self.labels[item]
        type, image = type_image
        if 'ACDC' in type:
            image, label = self._ACDC_downSample(image, label)
            label[label == 1] = 0
        else:
            image = self._downSample(image)
            label = self._downSample(label)

        if type == 'M&M':
            label[label == 3] = 0
        label[label > 0] = 1

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
        return image, label

    def __len__(self):
        return len(self.images)