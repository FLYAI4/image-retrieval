'''
model pth 파일 저장하는 코드
'''

import torch
import torchvision.models as models
import torch.nn as nn


resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
resnet = torch.nn.Sequential(*list(resnet.children())[:-1])
torch.save(resnet, 'resnet50_model.pth')

vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_FEATURES)
vgg = torch.nn.Sequential(*list(vgg.children())[:-1])
flatten = nn.Flatten()
vgg.add_module("Flatten", flatten)
torch.save(vgg, 'vgg16_model.pth')
