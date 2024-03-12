import matplotlib.pyplot as plt
import numpy as np
import glob
import os
import random
from PIL import Image
import cv2

np.set_printoptions(threshold=np.inf)

def down(image, label):
    img = cv2.resize(image, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_LINEAR)
    lb = cv2.resize(label, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_NEAREST)
    return img, lb

root = './'
all_packets = glob.glob(os.path.join(root, 'image', '*.jpg'))
for i, packet in enumerate(all_packets):
    # img = cv2.imread(packet)
    # lb = cv2.imread(packet.replace('image', 'mask'))
    # print(img.shape)
    # img, lb = down(img, lb)
    # print(img.shape)
    # cv2.imwrite(packet, img)
    # cv2.imwrite(packet.replace('image', 'mask'), lb)
    all_packets[i] = os.path.basename(packet).replace('.jpg', '')

print(all_packets)

train_packets = []
test_packets = []
val_packets = []

train_flag = random.sample(range(0, 40), 32)
val_flag = random.sample(train_flag, 8)
test_flag = list(set(range(0, 40)) - set(train_flag))
train_flag = list(set(train_flag) - set(val_flag))

for i in train_flag:
    train_packets.append(all_packets[i])

for i in test_flag:
    test_packets.append(all_packets[i])

for i in val_flag:
    val_packets.append(all_packets[i])

random.shuffle(train_packets)
random.shuffle(test_packets)
random.shuffle(val_packets)

print(train_packets)

train_paths = []
for packet in train_packets:
    paths = glob.glob(os.path.join(root, 'image_patchs', packet, '*.jpg'))
    train_paths.extend(paths)

test_paths = []
for packet in test_packets:
    paths = glob.glob(os.path.join(root, 'image_patchs', packet, '*.jpg'))
    test_paths.extend(paths)

val_paths = []
for packet in val_packets:
    paths = glob.glob(os.path.join(root, 'image_patchs', packet, '*.jpg'))
    val_paths.extend(paths)

random.shuffle(train_paths)
random.shuffle(test_paths)
random.shuffle(val_paths)

np.save('train_path.npy', train_paths)
np.save('test_path.npy', test_paths)
np.save('val_path.npy', val_paths)


