#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 19:21:08 2021

@author: mateusz
"""
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from torchvision import transforms


from torch.utils.data import DataLoader

from dataloader.FacialDetection import FacialDetectionDataset, my_collate
from networks.fcn import FCN2D, Pooled_FCN2D
from utils.image_processing import convert_tensor_to_im

#%% Find GPU
cuda = True if torch.cuda.is_available() else False
device = torch.device('cuda') if cuda else torch.device('cpu')

#%% Hyperparameters
BATCH_SIZE_TRAIN = 32
BATCH_SIZE_VAL = 5
EPOCHS = 100
CALLBACK_FREQ = 5

#%% Dataloader

transform_func = transforms.Compose([transforms.ToTensor(),
                                 transforms.Normalize([0.5], [0.5])])

data = FacialDetectionDataset('data/training.csv',
                              transform=transform_func,
                              target_transform=transform_func)

train_data, val_data = torch.utils.data.random_split(data, 
                                                     [len(data)-20, 20])

train_loader = DataLoader(dataset=train_data, 
                          batch_size=BATCH_SIZE_TRAIN, 
                          shuffle=True, 
                          num_workers=0)

val_loader = DataLoader(dataset=val_data, 
                        batch_size=BATCH_SIZE_VAL, 
                        shuffle=True, 
                        num_workers=0)

#%% Preparation for training
net = Pooled_FCN2D().to(device)

criterion = nn.MSELoss().to(device)
optim = torch.optim.Adam(net.parameters(), lr=0.0002, 
                                   betas=(0.5, 0.999))
losses_train = []
losses_val = []

for epoch in range(EPOCHS):
    net.train()
    for ii, (im, label) in enumerate(train_loader):
        optim.zero_grad()
        output = net(im.to(device))
        loss_train = criterion(output, label.to(device))
        loss_train.backward()
        optim.step()

    if epoch % CALLBACK_FREQ == 0:
        net.eval()
        val_im, val_label = next(iter(val_loader)) 
        output = net(val_im.to(device))
        
        loss_val = criterion(output, val_label.to(device))
        losses_val.append(loss_val.item())

        plt.figure(figsize=(5, 5))
        readable_output = np.sum(output[0].detach().cpu().numpy(),axis=0,
                                 keepdims=True)
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 10))
        ax1.imshow(convert_tensor_to_im(val_im[0]))
        ax2.imshow(convert_tensor_to_im(val_label[0], reduce_channel_dim=True))
        ax3.imshow(convert_tensor_to_im(output[0], reduce_channel_dim=True))
        plt.show()

    losses_train.append(loss_train.item())
    
torch.save(net.state_dict(), 'model.pth')
torch.save(optim.state_dict(), 'optimizer.pth')






