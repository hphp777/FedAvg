import torchvision.models as models
import torch.nn as nn
import torch
import os
import cv2
import PIL
import torchvision
import torchvision.transforms as transforms
import datetime
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib
from torchvision.utils import make_grid, save_image
import shutil
from glob import glob
import config
from datasets import XRaysTestDataset

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = models.resnet50(pretrained=True)
model.eval()
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 15) # 15 output classes 
model.to(device)

loss_fn = nn.BCEWithLogitsLoss().to(device)

imgs = glob("C:/Users/hb/Desktop/data/ChestX-ray14_Client_Data/test/*")
data_dir = "C:/Users/hb/Desktop/data/archive"

XRayTest_dataset = XRaysTestDataset(data_dir, transform = config.transform)
test_loader = torch.utils.data.DataLoader(XRayTest_dataset, batch_size = 1, shuffle = not True)

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

for i in range(len(imgs)):

    PATH = "C:/Users/hb/Desktop/code/FedAvg/models/server/CZ/server_5.pth"
    model.load_state_dict(torch.load(PATH))

    # final conv layer name 
    finalconv_name = 'layer4'

    # activations
    feature_blobs = []

    # gradients
    backward_feature = []

    # output으로 나오는 feature를 feature_blobs에 append하도록
    def hook_feature(module, input, output):
        feature_blobs.append(output.cpu().data.numpy())
    

    # Grad-CAM
    def backward_hook(module, input, output):
        backward_feature.append(output[0])
    
    # for name, module in model.named_children():
    #     if name == finalconv_name:
    #         for sub_name, sub_module in module[len(module)-1].named_children():
    #             if sub_name == 'conv2':
    #                 print(sub_module)
                    
    # model = model.layer4[2].conv3
    # print(model)
    # model.register_forward_hook(hook_feature)
    # model.register_backward_hook(backward_hook)
    # print(model._modules.get(finalconv_name))
    # model._modules.get(finalconv_name).register_forward_hook(hook_feature)
    # model._modules.get(finalconv_name).register_backward_hook(backward_hook)

    model.layer4[2].conv3.register_forward_hook(hook_feature)
    model.layer4[2].conv3.register_backward_hook(backward_hook)

    # get the softmax weight
    params = list(model.parameters())
    weight_softmax = np.squeeze(params[-2].cpu().detach().numpy()) # [1000, 512]


    pil_img = cv2.imread(imgs[i])
    
    target = XRayTest_dataset[i][1]
    target.to(device)
    # print(target[1])

    normalizer = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    torch_img = torch.from_numpy(np.asarray(pil_img)).permute(2, 0, 1).unsqueeze(0).float().div(255).cuda()
    torch_img = F.interpolate(torch_img, size=(1024, 1024), mode='bilinear', align_corners=False) # (1, 3, 224, 224)
    normed_torch_img = normalizer(torch_img)
    save_image(pil_img, 'result/img'+ str(i) + '.png')

    current_time = datetime.datetime.now() + datetime.timedelta(hours= 9)
    current_time = current_time.strftime('%Y-%m-%d-%H:%M')

    saved_loc = 'result'

    img_name = str(i) + '.jpg'
    sigmoid = torch.nn.Sigmoid()
    # Prediction
    logit = model(normed_torch_img)
    h_x = F.softmax(logit, dim=1).data.squeeze() # softmax 적용

    preds = sigmoid(logit).cpu().detach().numpy()
    preds[preds>0.3] = 1
    preds[preds<=0.3] = 0
    print("Predicted " + str(i) + " : " , preds)
    probs, idx = h_x.sort(0, True)


# ============================= #
# ==== Grad-CAM main lines ==== #
# ============================= #

    # score = loss_fn(logit[0].to(device), target.to(device))
    score = logit[:, idx[0]].squeeze() # 예측값 y^c
    score.backward(retain_graph = True) # 예측값 y^c에 대해서 backprop 진행

    activations = torch.Tensor(feature_blobs[0]).to(device) # (1, 512, 7, 7), forward activations
    gradients = backward_feature[0] # (1, 512, 7, 7), backward gradients
    b, k, u, v = gradients.size()

    alpha = gradients.view(b, k, -1).mean(2) # (1, 512, 7*7) => (1, 512), feature map k의 'importance'
    weights = alpha.view(b, k, 1, 1) # (1, 512, 1, 1)

    grad_cam_map = (weights*activations).sum(1, keepdim = True) # alpha * A^k = (1, 512, 7, 7) => (1, 1, 7, 7)
    grad_cam_map = F.relu(grad_cam_map) # Apply R e L U
    grad_cam_map = F.interpolate(grad_cam_map, size=(1024, 1024), mode='bilinear', align_corners=False) # (1, 1, 224, 224)
    map_min, map_max = grad_cam_map.min(), grad_cam_map.max()
    grad_cam_map = (grad_cam_map - map_min).div(map_max - map_min).data # (1, 1, 224, 224), min-max scaling
    
    # grad_cam_map.squeeze() : (224, 224)
    grad_heatmap = cv2.applyColorMap(np.uint8(255 * grad_cam_map.squeeze().cpu()), cv2.COLORMAP_JET) # (224, 224, 3), numpy 
    grad_heatmap = torch.from_numpy(grad_heatmap).permute(2, 0, 1).float().div(255) # (3, 244, 244)
    b, g, r = grad_heatmap.split(1)
    grad_heatmap = torch.cat([r, g, b]) # (3, 244, 244), opencv's default format is BGR, so we need to change it as RGB format.

    grad_result = grad_heatmap + torch_img.cpu() # (1, 3, 244, 244)
    grad_result = grad_result.div(grad_result.max()).squeeze() # (3, 244, 244)

    save_img =  'result/'+ str(i) + '_result.png'

    save_image(grad_result, save_img)

# image_list = []

# image_list.append(torch.stack([torch_img.squeeze().cpu(), grad_heatmap, grad_result], 0)) # (3, 3, 244, 244)

# images = make_grid(torch.cat(image_list, 0), nrow=3)

# output_dir = 'outputs'
# os.makedirs(output_dir, exist_ok=True)
# output_name = img_name
# output_path = os.path.join(output_dir, output_name)

# save_image(images, output_path)
# PIL.Image.open(output_path)