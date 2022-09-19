import torch
import pandas as pd
import os
from glob import glob
import shutil
import math
import numpy as np
import matplotlib.pyplot as plt

alpha = 0.5
c_num = 10

ratios = []
pathes = []

for i in range(10):
    os.makedirs('C:/Users/hb/Desktop/data/CIFAR100_Client/C' + str(i))


for i in range(100):
    ratios.append(np.random.dirichlet(np.repeat(alpha, c_num)))

classes = [x for x in glob('C:/Users/hb/Desktop/data/CIFAR100/train/*')]
class_names = []

for i in range(100):
    cc = classes[i].split("\\")
    class_names.append(cc[1])

for i in range(100):
    path = 'C:/Users/hb/Desktop/Data/CIFAR100/train/' + class_names[i] + "/"
    pathes.append(path)


mat = np.array(ratios)
fig = plt.figure(figsize=(8,6))
plt.pcolormesh(mat,cmap=plt.cm.Blues)
plt.xticks(range(10))
plt.xlabel('Client ID')
plt.ylabel('Class ID')
plt.colorbar()
plt.show()

for c in range(len(pathes)):
    index = 0
    total_img_num = 500

    for client in range(10):
        if client == 9:
            for img in range(index, total_img_num):
                img_name = class_names[c] + '_' + str(img) + '.png'
                src = pathes[c] + img_name
                dst = 'C:/Users/hb/Desktop/Data/CIFAR100_Client/C' + str(client) + '/' + img_name
                shutil.copy(src,dst)
        else:
            for img in range(index, math.floor(index + ratios[c][client]*total_img_num)):
                img_name = class_names[c] + '_' + str(img) + '.png'
                src = pathes[c] + img_name
                dst = 'C:/Users/hb/Desktop/Data/CIFAR100_Client/C' + str(client) + '/' + img_name
                shutil.copy(src,dst)
        index = index + math.floor(ratios[c][client]*total_img_num)