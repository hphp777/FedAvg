import torch
import pandas as pd
import os
from glob import glob
import shutil
import math
import numpy as np
import matplotlib.pyplot as plt
import random

alpha = 0.5
c_num = 2

ratios = []
pathes = []
dst_pathes1 = []
dst_pathes2 = []

classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# for i in range(10):
#     os.makedirs('C:/Users/hb/Desktop/data/CIFAR10_Client/C' + str(i))
#     os.makedirs('C:/Users/hb/Desktop/data/CIFAR1100_Client/C' + str(i))

for i in range(10):
    # ratios.append(np.random.dirichlet(np.repeat(alpha, c_num)))
    ratios.append([0.75,0.25])

classes = [x for x in glob('C:/Users/hb/Desktop/data/CIFAR10/train/*')]
class_names = []

for i in range(10):
    cc = classes[i].split("\\")
    class_names.append(cc[1])

# for i in range(10):
#     os.makedirs('C:/Users/hb/Desktop/data/CIFAR10_shuffle/' + class_names[i])

for i in range(10):
    path = 'C:/Users/hb/Desktop/Data/CIFAR10_shuffle/' + class_names[i] + "/"
    pathes.append(path)

for i in range(10):
    path = 'C:/Users/hb/Desktop/Data/CIFAR10_1/Split1/' + class_names[i] + "/"
    os.makedirs(path)
    dst_pathes1.append(path)

for i in range(10):
    path = 'C:/Users/hb/Desktop/Data/CIFAR10_1/Split2/' + class_names[i] + "/"
    dst_pathes2.append(path)

mat = np.array(ratios)
plt.imshow(mat, cmap = plt.cm.Blues)
plt.xticks(range(10))
plt.xlabel('Client ID')
plt.ylabel('Class ID')
plt.yticks(range(10))
plt.colorbar()
plt.show()

for c in range(len(pathes)):
    index = 0
    total_img_num = 5000
    total_path = glob(pathes[c] + '*.png')
    random.shuffle(total_path)

    for client in range(c_num):
        if client == 1:
            for img in range(index, total_img_num):
                img_name = class_names[c] + '_' + str(img) + '.png'
                src = total_path[img]
                dst = dst_pathes2[c] + img_name
                shutil.copy(src,dst)
        else:
            for img in range(index, math.floor(index + ratios[c][client]*total_img_num)):
                img_name = class_names[c] + '_' + str(img) + '.png'
                src = total_path[img]
                dst = dst_pathes1[c] + img_name
                shutil.copy(src,dst)

        index = index + math.floor(ratios[c][client]*total_img_num)

