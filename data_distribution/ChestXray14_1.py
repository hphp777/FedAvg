from cgi import test
import torch
import pandas as pd
import os
from glob import glob
import shutil

index = 0
all_xrays_df = pd.read_csv('./data_list/Pneumothorax.csv') # 

my_file = open("test_list.txt", "r")
data = my_file.read()
testdata = data.replace('\n', ' ').split(" ")

print(testdata)

# folder path 
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

pathes = [Atelectasis_path, Cardiomegaly_path, Consolidation_path, Edema_path, Effusion_path, Emphysema_path, Fibrosis_path, Hernia_path, Infiltration_path, Mass_path, Nodule_path, Pleural_Thickening_path, Pneumothorax_path, Pneumonia_path, Nofinding_path]
diseases = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion', 'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass', 'Nodule', 'Pleural_Thickening', 'Pneumothorax', 'Pneumonia', 'NoFinding']

pathes = [Nofinding_path]
diseases = ['NoFinding']

# for i in range(len(diseases)):
#     path = 'C:/Users/hb/Desktop/data/ChestX-ray14/' + diseases[i]
#     os.makedirs(path)


for i in range(len(pathes)):
    index = 0
    all_xrays_df = pd.read_csv('./data_list/' + diseases[i] + '.csv')
    for row in all_xrays_df.itertuples():
        if row[2] in testdata:
            # os.remove(row[14])
            continue
        else:
            src = row[14]
            img_name =  diseases[i] + '_' + str(index) + '.png' # 
            dst = pathes[i] + img_name # 
            shutil.copy(src,dst)
            index += 1

aa = glob('C:/Users/hb/Desktop/data/ChestX-ray14/*/*.png')
print(len(aa))

# for i in range(len(pathes)):
#     index = 0
#     all_xrays_df = pd.read_csv('./data_list/' + diseases[i] + '.csv')   
#     print(len(all_xrays_df.itertuples()))










