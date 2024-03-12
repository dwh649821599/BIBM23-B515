import matplotlib.pyplot as plt
import numpy as np
import glob
import os
import random
from PIL import Image
import cv2

np.set_printoptions(threshold=np.inf)

root = './'
all_packets = glob.glob(os.path.join(root, 'image', '*.jpg'))
for i, packet in enumerate(all_packets):
    all_packets[i] = os.path.basename(packet).replace('.jpg', '')

print(all_packets)

train_packets = []
test_packets = []
val_packets = []

train_flag = random.sample(range(0, 28), 23)
val_flag = random.sample(train_flag, 5)
test_flag = list(set(range(0, 28)) - set(train_flag))
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

print(len(train_paths))

np.save('train_path.npy', train_paths)
np.save('test_path.npy', test_paths)
np.save('val_path.npy', val_paths)

