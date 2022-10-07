import argparse
import os, pdb, sys, glob, time
import numpy as np
import pandas as pd
from tqdm import tqdm
import cv2
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
import torchvision.models as models 
from model.resnet import ResNet50_fedalign, ResNet50

# import custom dataset classes
from datasets import XRaysTrainDataset  ,ChestXLoader
from datasets import XRaysTestDataset

# import neccesary libraries for defining the optimizers
import torch.optim as optim

from trainer import fit
import config
from base import client, server

import warnings

import importlib
importlib.reload(models)

warnings.filterwarnings(action='ignore')

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# model = ResNet50.resnet56() ####
# model = ResNet50_fedalign.resnet56()
model = models.resnet50(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 15) # 15 output classes 
model.to(device)

c_num = 5
com_round = 20

data_dir = "C:/Users/hb/Desktop/data/archive"

XRayTrain_dataset = XRaysTrainDataset(data_dir, transform = config.transform)
ratios = np.round(np.random.dirichlet(np.repeat(10, c_num))*len(XRayTrain_dataset)).astype(int)
if sum(ratios) > len(XRayTrain_dataset):
    ratios[4] -= (sum(ratios) - len(XRayTrain_dataset))
else:
    ratios[4] += (len(XRayTrain_dataset) - sum(ratios))

data0,data1,data2,data3,data4 = torch.utils.data.random_split(XRayTrain_dataset, ratios)
central_data = XRaysTrainDataset(data_dir, transform = config.transform)

central_server = server()
client0 = client(0,central_data)
# client1 = client(1,data1)
# client2 = client(2,data2)
# client3 = client(3,data3)
# client4 = client(4,data4)

clients = [client0]
weights = [0] * 5
weight = model.state_dict()
server_auc = []
server_acc = []
best_acc = 0
best_auc = 0

total_data_num = 0

# for i in range(c_num):
#     total_data_num += len(clients[i].dataset)

def draw_auc():

    for i in range(2,21):
        path = "C:/Users/hb/Desktop/code/FedAvg/models/server/FedAvg/server_" + str(i) + ".pth"
        model = torch.load(path)
        auc ,acc= central_server.test(model)
        server_auc.append(auc)
        server_acc.append(acc)

    plt.plot(range(len(server_acc)), server_acc)
    plt.plot(range(len(server_auc)), server_auc)
    plt.savefig('./results/FedAvg_acc_auc.png')
    plt.clf()

def FL():

    print("\nCommunication Round 1")

    for i in range(c_num):
        weights[i] = clients[i].train()

    cw = []
    for i in range(c_num):
        cw.append(len(clients[i].dataset) / total_data_num)

    for key in weights[0]:
        weight[key] = sum([weights[i][key] * cw[i] for i in range(c_num)]) 

    # Test
    auc ,acc= central_server.test(weight)
    best_acc = acc
    best_auc =auc
    server_auc.append(auc)
    server_acc.append(acc)

    for r in range(2, com_round+1):

        print("\nCommunication Round " + str(r))

        for i in range(c_num):
            weights[i] = clients[i].train(updated=True, weight=weight)

        for key in weights[0]:
            weight[key] = sum([weights[i][key] * cw[i] for i in range(c_num)]) 

        torch.save(weight, 'C:/Users/hb/Desktop/code/FedAvg/models/server/FedAlign/server_' + str(r) + '.pth' )

        # Test
        auc, acc = central_server.test(weight)
        if auc > best_auc:
            best_auc = auc
        if acc > best_acc:
            best_acc = acc
        server_auc.append(auc)
        server_acc.append(acc)

    print("AUCs : ", server_auc)
    print("Best AUC: ", best_auc)
    print("Best Acc: ", best_acc)

    return server_auc,server_acc

def CZ():
    
    best_acc = 0
    best_auc = 0

    for i in range(20):

        print("Epoch{}".format(i))
        weight = clients[0].train()
        auc, acc = central_server.test(weight)

        torch.save(weight, 'C:/Users/hb/Desktop/code/FedAvg/models/server/CZ/server_' + str(i) + '.pth' )
    
        if auc > best_auc:
            best_auc = auc
        if acc > best_acc:
            best_acc = acc
    
        server_auc.append(auc)
        server_acc.append(acc)
    
    print("AUCs : ", server_auc)
    print("Best AUC: ", best_auc)
    print("Best Acc: ", best_acc)

if __name__ == '__main__':
    CZ()
    # draw_auc()