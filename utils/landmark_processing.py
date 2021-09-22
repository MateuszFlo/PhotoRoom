#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 19:04:14 2021

@author: mateusz
"""
import numpy as np

def apply_gaussian(x0, y0, sigma, height, width):
        x = np.arange(0, width, 1)
        y = np.arange(0, height, 1)[:, np.newaxis]
        return np.exp(-((x-x0)**2 + (y-y0)**2) / (2*sigma**2))
    
    
def create_heatmap(landmarks, sigma, height, width):
        nb_landmarks = landmarks.shape[0]
        heatmap = np.empty((nb_landmarks, height, width), 
                           dtype = np.float32)
        for i in range(nb_landmarks):
            if not np.isnan(landmarks[i]).any():
                heatmap[i,:,:] = apply_gaussian(landmarks[i][0],
                                           landmarks[i][1],
                                           sigma ,width, height)
            else:
                #return None
                heatmap[i,:,:] = np.zeros(heatmap.shape[1:])
        return heatmap
    
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    landmark = np.array([[66, 39]])
    heatmap = create_heatmap(landmark, 5, 96, 96).transpose(1,2,0)
    plt.imshow(heatmap)
    
def flip(image, label, proba= 0.5):
    p = np.random.uniform()
    if p > proba:
        return image, label
    
    
    image = np.fliplr(image)
    label = np.fliplr(label)
    # Left eye is now the right eye, so its index needs to be changed 
    # in the labels
    label_new_indexes = [1, 0, 4, 5, 2, 3, 8, 9, 6, 7, 10, 12, 11, 13, 14]
    label = label[...,label_new_indexes]
    
    return np.ascontiguousarray(image), np.ascontiguousarray(label)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from torchvision import transforms
    from dataloader.FacialDetection import FacialDetectionDataset
    
    transform_func = transforms.Compose([transforms.ToTensor(),
                                 transforms.Normalize([0.5], [0.5])])
    
    data = FacialDetectionDataset('data/training.csv')
    im, label = data[0]
    
    flipped_im, flipped_label = flip(im, label, proba=1)
    left_eye = label[...,:1]
    flipped_left_eye = flipped_label[...,:1]
    
    fig, ax = plt.subplots(2, 2, figsize=(15, 15))
    ax[0,0].imshow(im + left_eye)
    ax[0,1].imshow(flipped_im + flipped_left_eye)
    
    right_mouth_corner = label[...,-3:-2]
    flipped_right_mouth_corner = flipped_label[...,-3:-2]
    
    ax[1,0].imshow(im + right_mouth_corner)
    ax[1,1].imshow(flipped_im + flipped_right_mouth_corner)

    
    