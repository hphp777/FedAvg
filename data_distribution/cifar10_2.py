import os
from glob import glob
import shutil
import random

# Allocate image to its corresponding class folder

classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

for i in range(len(classes)):
    all_image_paths = [x for x in glob('C:/Users/hb/Desktop/data/from/train/' + classes[i] + '/*.png')]
    index = 0
    print(all_image_paths)
    for j in range(len(all_image_paths)):
        src = all_image_paths[j]
        dst = 'C:/Users/hb/Desktop/data/CIFAR10/train/' + classes[i] + '/' + classes[i] + '_' + str(j) + '.png'
        shutil.copy(src,dst)
        index += 1

for i in range(len(classes)):
    all_image_paths = [x for x in glob('C:/Users/hb/Desktop/data/from/test/' + classes[i] + '/*.png')]
    index = 0
    for j in range(len(all_image_paths)):
        src = all_image_paths[j]
        dst = 'C:/Users/hb/Desktop/data/CIFAR10/test/' + classes[i] + '/' + classes[i] + '_' + str(j) + '.png'
        shutil.copy(src,dst)
        index += 1

# Shuffle
for i in range(len(classes)):
    all_image_paths = [x for x in glob('C:/Users/hb/Desktop/data/CIFAR10/train/' + classes[i] + '/*.png')]
    index = 0
    random.shuffle(all_image_paths)
    for j in range(len(all_image_paths)):
        src = all_image_paths[j]
        dst = 'C:/Users/hb/Desktop/data/CIFAR10_shuffle/' + classes[i] + '/' + classes[i] + '_' + str(j) + '.png'
        shutil.copy(src,dst)
        index += 1
