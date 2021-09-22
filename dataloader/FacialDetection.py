#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 19:14:58 2021

@author: mateusz
"""
import numpy as np
import pandas as pd

from torch.utils.data import Dataset, dataloader
from torchvision import transforms

from utils.landmark_processing import create_heatmap, flip
from utils.image_processing import convert_tensor_to_im

class FacialDetectionDataset(Dataset):
    def __init__(self, annotations_file, transform=None, target_transform=None):
        self.data = pd.read_csv(annotations_file)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Process image to be compliant with ToTensor()
        image = np.array(np.matrix(self.data.iloc[idx, -1]) / 255)
        image = image.astype(np.float32).reshape(96,96)
        image = image[np.newaxis, ...].repeat(3,axis=0)
        image = image.transpose(1,2,0)
        
        labels = np.array(self.data.iloc[idx, :-1]).astype(np.float32)
        labels = labels.reshape(-1, 2)
        label_heatmap = create_heatmap(labels, 3, 96, 96)
        
        # Tried to remove the data that does not contain all landmarks from
        # training but got ErrorType with my_collate
# =============================================================================
#         if label_heatmap is None:
#             return None, None
# =============================================================================
        
        label_heatmap = label_heatmap.transpose(1,2,0)
        
        # Augmentation
        image, label_heatmap = flip(image, label_heatmap, proba=0.5)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label_heatmap = self.target_transform(label_heatmap)
        return image, label_heatmap
    
def my_collate(batch):
    batch = filter(lambda img: img is not None, batch)
    return dataloader.default_collate(list(batch))
    
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    transform_func = transforms.Compose([transforms.ToTensor(),
                                 transforms.Normalize([0.5], [0.5])])
    
    data = FacialDetectionDataset('data/training.csv',
                                  transform = transform_func,
                                  target_transform=transform_func)
    im, label = data[-5]
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 10))
    ax1.imshow(convert_tensor_to_im(im))
    ax2.imshow(convert_tensor_to_im(label, reduce_channel_dim=True))
    ax3.imshow(convert_tensor_to_im(label, reduce_channel_dim=True) + 
               convert_tensor_to_im(im))