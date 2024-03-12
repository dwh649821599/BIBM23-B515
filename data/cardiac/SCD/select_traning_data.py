import numpy as np
import glob
import os
import random

root = './'
all_packets = glob.glob(os.path.join(root, 'SC*'))
for i, packet in enumerate(all_packets):
    all_packets[i] = os.path.basename(packet)

train_packets = []
test_packets = []
val_packets = []

train_flag = random.sample(range(0, 45), 36)
val_flag = random.sample(train_flag, 4)
test_flag = list(set(range(0, 45)) - set(train_flag))
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

train_paths = []
for packet in train_packets:
    paths = glob.glob(root + '/{}*/image*'.format(packet))
    train_paths.extend(paths)

test_paths = []
for packet in test_packets:
    paths = glob.glob(root + '/{}*/image*'.format(packet))
    test_paths.extend(paths)

val_paths = []
for packet in val_packets:
    paths = glob.glob(root + '/{}*/image*'.format(packet))
    val_paths.extend(paths)

print(train_paths)

np.save('train_path1.npy', train_paths)
np.save('test_path1.npy', test_paths)
np.save('val_path1.npy', val_paths)

aa = 1
