import argparse
import os, pdb, sys, glob, time
from random import shuffle
import numpy as np
import pandas as pd
from tqdm import tqdm
import cv2
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
import torchvision.models as models 
import torch.nn as nn
from model.resnet import ResNet50_fedalign, ResNet50

# import custom dataset classes
from datasets import GANData

# import neccesary libraries for defining the optimizers
import torch.optim as optim
import os
import csv

GANDataset = GANData()
GANLoader = torch.utils.data.DataLoader(GANDataset, batch_size = 256, shuffle = False)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

model = models.resnet50(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 15) # 15 output classes 
model.to(device)
model.load_state_dict(torch.load("C:/Users/hb/Desktop/code/FedAvg/models/server/CZ/server_1.pth"))

dataList = glob.glob("C:/Users/hb/Desktop/code/FedAvg/generated_img/*")



disease = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion', 'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass', 'No Finding', 'Nodule', 'Pleural_Thickening', 'Pneumonia', 'Pneumothorax']

sigmoid = torch.nn.Sigmoid()

f = open('GAN_df.csv','w', newline='')

for i, (img, label) in enumerate(GANLoader):

    img = img.to(device)

    print(label)

    out = model(img)
    preds = np.round(sigmoid(out).cpu().detach().numpy())
    
    # 처음 질병은 | 없이 쓰기

    # 두번째 질병은 | 와 함께 쓰기

    # 만약 아무것으로도 분류되지 않는 사진은 지우기
    
    # flag = 0
    # d = []
    # for c in range(len(preds)):
        
    #     if preds[c] == 1:
    #         d.append(disease[c])

    # if len(d) == 0:
    #     os.remove(dataList[i])

    
