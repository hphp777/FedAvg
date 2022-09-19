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

warnings.filterwarnings(action='ignore')

# model = ResNet50.resnet56() ####
# model = ResNet50_fedalign.resnet56()
model = models.resnet50(pretrained=True)

c_num = 5
com_round = 20

data_dir = "C:/Users/hb/Desktop/data/archive"

XRayTrain_dataset = XRaysTrainDataset(data_dir, transform = config.transform)
ratios = np.round(np.random.dirichlet(np.repeat(10, c_num))*len(XRayTrain_dataset)).astype(int)
if sum(ratios) > len(XRayTrain_dataset):
    ratios[4] -= (sum(ratios) - len(XRayTrain_dataset))
else:
    ratios[4] += (len(XRayTrain_dataset) - sum(ratios))

print("Client Data:  ",ratios)

data0,data1,data2,data3,data4 = torch.utils.data.random_split(XRayTrain_dataset, ratios)

central_server = server()
client0 = client(0,data0)
client1 = client(1,data1)
client2 = client(2,data2)
client3 = client(3,data3)
client4 = client(4,data4)

clients = [client0, client1, client2, client3,client4]
weights = [0] * 5
weight = model.state_dict()
server_auc = []


total_data_num = 0

for i in range(c_num):
    total_data_num += len(clients[i].dataset)

def draw_auc(aucs):
    plt.plot(range(com_round), aucs)
    plt.savefig('./result/Server_test_accuracy.png')
    plt.clf()

def CZ():

    print("Communication Round 1")

    for i in range(c_num):
        weights[i] = clients[i].train()

    cw = []
    for i in range(c_num):
        cw.append(len(clients[i].dataset) / total_data_num)

    for key in weights[0]:
        weight[key] = sum([weights[i][key] * cw[i] for i in range(c_num)]) 

    # Test
    auc = central_server.test(weight)
    server_auc.append(auc)

    for r in range(com_round-1):

        print("Communication Round " + str(r))

        for i in range(c_num):
            weights[i] = clients[i].train(updated=True, weight=weight)

        for key in weights[0]:
            weight[key] = sum([weights[i][key] * cw[i] for i in range(c_num)]) 

        torch.save(weight, 'C:/Users/hb/Desktop/code/FL_distribution_skew/models/server/server_' + str(r) + '.pth' )

        # Test
        auc = central_server.test(weight)
        server_auc.append(auc)

if __name__ == '__main__':
    CZ()