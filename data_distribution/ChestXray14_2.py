import torch
import pandas as pd
import os
from glob import glob
import shutil
import math
import numpy as np
import matplotlib.pyplot as plt

alpha = 20
c_num = 5

# Label skew for each client

Atelectasis_path = 'C:/Users/hb/Desktop/data/ChestX-ray14/Atelectasis/'
Cardiomegaly_path = 'C:/Users/hb/Desktop/data/ChestX-ray14/Cardiomegaly/'
Consolidation_path = 'C:/Users/hb/Desktop/data/ChestX-ray14/Consolidation/'
Edema_path = 'C:/Users/hb/Desktop/data/ChestX-ray14/Edema/'
Effusion_path = 'C:/Users/hb/Desktop/data/ChestX-ray14/Effusion/'
Emphysema_path = 'C:/Users/hb/Desktop/data/ChestX-ray14/Emphysema/'
Fibrosis_path = 'C:/Users/hb/Desktop/data/ChestX-ray14/Fibrosis/'
Hernia_path = 'C:/Users/hb/Desktop/data/ChestX-ray14/Hernia/'
Infiltration_path = 'C:/Users/hb/Desktop/data/ChestX-ray14/Infiltration/'
Mass_path = 'C:/Users/hb/Desktop/data/ChestX-ray14/Mass/'
Nodule_path = 'C:/Users/hb/Desktop/data/ChestX-ray14/Nodule/'
Pleural_Thickening_path = 'C:/Users/hb/Desktop/data/ChestX-ray14/Pleural_Thickening/'
Pneumothorax_path = 'C:/Users/hb/Desktop/data/ChestX-ray14/Pneumothorax/'
Pneumonia_path = 'C:/Users/hb/Desktop/data/ChestX-ray14/Pneumonia/'
Nofinding_path = 'C:/Users/hb/Desktop/data/ChestX-ray14/NoFinding/'

Atelectasis_ratio = np.random.dirichlet(np.repeat(alpha, c_num))
Cardiomegaly_ratio = np.random.dirichlet(np.repeat(alpha, c_num))
Consolidation_ratio = np.random.dirichlet(np.repeat(alpha, c_num))
Edema_ratio = np.random.dirichlet(np.repeat(alpha, c_num))
Effusion_ratio = np.random.dirichlet(np.repeat(alpha, c_num))
Emphysema_ratio = np.random.dirichlet(np.repeat(alpha, c_num))
Fibrosis_ratio = np.random.dirichlet(np.repeat(alpha, c_num))
Hernia_ratio = np.random.dirichlet(np.repeat(alpha, c_num))
Infiltration_ratio = np.random.dirichlet(np.repeat(alpha, c_num))
Mass_ratio = np.random.dirichlet(np.repeat(alpha, c_num))
Nodule_ratio = np.random.dirichlet(np.repeat(alpha, c_num))
Pleural_Thickening_ratio = np.random.dirichlet(np.repeat(alpha, c_num))
Pneumothorax_ratio = np.random.dirichlet(np.repeat(alpha, c_num))
Pneumonia_ratio = np.random.dirichlet(np.repeat(alpha, c_num))
Nofinding_ratio = np.random.dirichlet(np.repeat(alpha, c_num))

pathes = [Atelectasis_path, Cardiomegaly_path, Consolidation_path, Edema_path,  Effusion_path, Emphysema_path, Fibrosis_path, Hernia_path, Infiltration_path, Mass_path, Nodule_path, Pleural_Thickening_path, Pneumothorax_path, Pneumonia_path, Nofinding_path]
diseases = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion', 'Emphysema', 'Fibrosis', 'Hernia','Infiltration', 'Mass', 'Nodule', 'Pleural_Thickening', 'Pneumothorax', 'Pneumonia', 'Nofinding']
ratios = [Atelectasis_ratio, Cardiomegaly_ratio, Consolidation_ratio,  Edema_ratio, Effusion_ratio, Emphysema_ratio, Fibrosis_ratio, Hernia_ratio, Infiltration_ratio, Mass_ratio, Nodule_ratio, Pleural_Thickening_ratio, Pneumothorax_ratio, Pneumonia_ratio, Nofinding_ratio]

mat = np.array(ratios)
plt.pcolormesh(mat, cmap = plt.cm.Blues)
plt.xticks(range(c_num))
plt.xlabel('Client ID')
plt.ylabel('Class ID')
plt.yticks(range(len(pathes)))
plt.colorbar()
plt.show()

for i in range(c_num):
    path = 'C:/Users/hb/Desktop/Data/ChestX-ray14_Client_Data/C' + str(i)
    os.makedirs(path)

for disease in range(len(pathes)):
    index = 0
    all_image_paths = {os.path.basename(x): x for x in 
                   glob('C:/Users/hb/Desktop/Data/ChestX-ray14/' + diseases[disease] + '/*.png')}
    total_img_num = len(all_image_paths)
    print(total_img_num)

    for client in range(c_num):
        if client == 4:
            for img in range(index, total_img_num):
                img_name = diseases[disease] + '_' + str(img) + '.png'
                src = pathes[disease] + img_name
                dst = 'C:/Users/hb/Desktop/Data/ChestX-ray14_Client_Data/C' + str(client) + '/' + img_name
                shutil.copy(src,dst)
        else:
            for img in range(index, math.floor(index + ratios[disease][client]*total_img_num)):
                img_name = diseases[disease] + '_' + str(img) + '.png'
                src = pathes[disease] + img_name
                dst = 'C:/Users/hb/Desktop/Data/ChestX-ray14_Client_Data/C' + str(client) + '/' + img_name
                shutil.copy(src,dst)
        index = index + math.floor(ratios[disease][client]*total_img_num)