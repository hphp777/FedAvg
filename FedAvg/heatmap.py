import os
import argparse

import cv2
import torch
import numpy as np
from torch.nn import functional as F
import torchvision.transforms as transforms

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
import config as cc

from glob import glob


def normalize(tensor, mean, std):
    if not tensor.ndimension() == 4:
        raise TypeError('tensor should be 4D')

    mean = torch.FloatTensor(mean).view(1, 3, 1, 1).expand_as(tensor).to(tensor.device)
    std = torch.FloatTensor(std).view(1, 3, 1, 1).expand_as(tensor).to(tensor.device)

    return tensor.sub(mean).div(std)


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        return self.do(tensor)
    
    def do(self, tensor):
        return normalize(tensor, self.mean, self.std)

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

def create_cam(config):

    imgs = glob("C:/Users/hb/Desktop/data/ChestX-ray14_Client_Data/test/*")
    finalconv_name = 'conv2'
    
    data_dir = "C:/Users/hb/Desktop/data/archive"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    XRayTest_dataset = XRaysTrainDataset(data_dir, transform = cc.transform)
    test_loader = torch.utils.data.DataLoader(XRayTest_dataset, batch_size = 1, shuffle = not True)

    model = models.resnet50(pretrained=True)
    model.eval()
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 15) # 15 output classes 
    model.to(device)
    PATH = "C:/Users/hb/Desktop/code/FedAvg/models/server/CZ/server_19.pth"
    model.load_state_dict(torch.load(PATH))

    # hook
    feature_blobs = []
    def hook_feature(module, input, output):
        feature_blobs.append(output.cpu().data.numpy())

    model.layer4[2].conv3.register_forward_hook(hook_feature)
    # model._modules.get(finalconv_name).register_forward_hook(hook_feature)
    params = list(model.parameters())
    # get weight only from the last layer(linear)
    weight_softmax = np.squeeze(params[-2].cpu().data.numpy())

    def returnCAM(feature_conv, weight_softmax, class_idx):
        size_upsample = (config.img_size, config.img_size)
        _, nc, h, w = feature_conv.shape
        output_cam = []
        cam = weight_softmax[class_idx].dot(feature_conv.reshape((nc, h*w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam) - 8
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))
        return output_cam
    
    for i in range(len(imgs)):

        pil_img = cv2.imread(imgs[i])

        normalizer = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        torch_img = torch.from_numpy(np.asarray(pil_img)).permute(2, 0, 1).unsqueeze(0).float().div(255).cuda()
        torch_img = F.interpolate(torch_img, size=(1024, 1024), mode='bilinear', align_corners=False) # (1, 3, 224, 224)
        normed_torch_img = normalizer(torch_img)

        image_tensor = normed_torch_img.to(device)
        logit= model(image_tensor)
        h_x = F.softmax(logit, dim=1).data.squeeze()
        probs, idx = h_x.sort(0, True)
        # print("True label : %d, Predicted label : %d, Probability : %.2f" % (label.item(), idx[0].item(), probs[0].item()))
        CAMs = returnCAM(feature_blobs[0], weight_softmax, [idx[0].item()])
        # img = cv2.imread(os.path.join(config.result_path, 'img%d.png' % (i + 1)))
        # height, width, _ = normed_torch_img.shape
        heatmap = cv2.applyColorMap(cv2.resize(CAMs[0], (1024, 1024)), cv2.COLORMAP_JET)
        result = heatmap * 0.3 + cv2.imread(imgs[i]) * 0.5
        cv2.imwrite(os.path.join(config.result_path, 'cam%d.png' % (i + 1)), result)
        if i + 1 == config.num_result:
            break
        feature_blobs.clear()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--result_path', type=str, default='./result')
    parser.add_argument('--img_size', type=int, default=1024)
    parser.add_argument('--num_result', type=int, default=1000)

    config = parser.parse_args()
    print(config)

    create_cam(config)