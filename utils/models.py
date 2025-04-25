'''
Due to lack or resources such as GPU and Powerful CPU, I have done all the training
using Kaggle and Colab. After, training the models i have downloaded the weights and 
then loaded in my local repository. This code snippet loads both YOLO, and UNet trained
models. 
'''

import torch
from ultralytics import YOLO
from Models.Unet import UNet

def load_unet_model(path):
    model = UNet()
    model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    model.eval()
    return model

def load_yolo_model(path):
    return YOLO(path)
