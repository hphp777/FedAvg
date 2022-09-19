import os
from glob import glob
import shutil

# Allocate image to its corresponding class folder

classes = [x for x in glob('C:/Users/hb/Desktop/data/from/train/*')]
class_names = []

for i in range(len(classes)):
    class_name = classes[i].split("\\")[1]
    print(class_name)
    class_names.append(class_name)
    os.makedirs('C:/Users/hb/Desktop/data/CIFAR100/train/' + class_name)
    os.makedirs('C:/Users/hb/Desktop/data/CIFAR100/test/' + class_name)

print(class_names)

for i in range(len(class_names)):
    all_image_paths = [x for x in glob('C:/Users/hb/Desktop/data/from/train/' + class_names[i] + '/*.png')]
    index = 0
    for j in range(len(all_image_paths)):
        src = all_image_paths[j]
        dst = 'C:/Users/hb/Desktop/data/CIFAR100/train/' + class_names[i] + '/' + class_names[i] + '_' + str(j) + '.png'
        shutil.copy(src,dst)
        index += 1

for i in range(len(class_names)):
    all_image_paths = [x for x in glob('C:/Users/hb/Desktop/data/from/test/' + class_names[i] + '/*.png')]
    index = 0
    for j in range(len(all_image_paths)):
        src = all_image_paths[j]
        dst = 'C:/Users/hb/Desktop/data/CIFAR100/test/' + class_names[i] + '/' + class_names[i] + '_' + str(j) + '.png'
        shutil.copy(src,dst)
        index += 1