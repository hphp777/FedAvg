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

import efficientnet.tfkeras as efn

import random
import cv2
from keras import backend as K
from keras.preprocessing import image
from sklearn.metrics import roc_auc_score, roc_curve
from tensorflow.compat.v1.logging import INFO, set_verbosity

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

import os
from glob import glob

from tensorflow.keras.preprocessing import image

import cv2
import csv

from keras.preprocessing.image import ImageDataGenerator
from keras.applications.densenet import DenseNet121
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from keras.models import load_model

# from tensorflow.keras.applications import DenseNet121
import tensorflow as tf
import tensorflow.keras.layers as L
# import tensorflow.keras.layers as Layers

# model import
from keras.applications.mobilenet import MobileNet
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.applications.nasnet import NASNetLarge
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.xception import Xception
from keras.applications.densenet import DenseNet201

# model
from keras.layers import GlobalAveragePooling2D, Dense, Dropout, Flatten
from keras.models import Sequential
from keras import optimizers, callbacks, regularizers

GANDataset = GANData()
GANLoader = torch.utils.data.DataLoader(GANDataset, batch_size = 256, shuffle = False)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
dataList = glob("C:/Users/hb/Desktop/code/FedAvg/generated_img/*")
disease = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion', 'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass', 'No Finding', 'Nodule', 'Pleural_Thickening', 'Pneumonia', 'Pneumothorax']
IMAGE_SIZE=[320, 320]

base_model = efn.EfficientNetB1(
                    input_shape = (*IMAGE_SIZE, 3), 
                    include_top = False, 
                    # weights = None
                    weights='imagenet'
                    )
model = Sequential()
model.add(base_model)
model.add(GlobalAveragePooling2D())
model.add(Dropout(0.5))
model.add(Dense(1024))
model.add(Dropout(0.5))
model.add(Dense(14, activation = 'sigmoid'))
model.compile(
    optimizer=tf.keras.optimizers.Adam( learning_rate=1e-4, amsgrad=False), 
    loss = 'binary_crossentropy',
    metrics = ['binary_accuracy']
)
model.load_weights('./efficent_net_b1_trained_weights.h5')
pathes = glob('C:/Users/hb/Desktop/data/ChestX-ray14_Client_Data/test/*.png')

df = pd.DataFrame(columns =['Image Index','Patient ID','Cardiomegaly','Emphysema','Effusion','Hernia','Infiltration','Mass','Nodule','Atelectasis','Pneumothorax','Pleural_Thickening','Pneumonia','Fibrosis','Edema','Consolidation','FilePath'])

for i in range(len(pathes)):

    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]
    img = np.array(image.load_img(pathes[i], target_size=(320, 320))).astype('float64')
    img -= mean
    img /= std
    img = np.expand_dims(img, axis=0)
    pred = model.predict(img)

    id = i
    name = pathes[i].split('\\')[1]

    item = [name,i]
    for d in range(14):
        item.append(int(pred[0][d])) 
    item.append(pathes[i])

    df.loc[i] = item
    
df.to_csv('pggan_main_df.csv')

    
