#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 19:53:17 2021

@author: mateusz
"""
import numpy as np

def convert_tensor_to_im(tensor, reduce_channel_dim=False):
    """ Display a tensor as an image. """
    image = tensor.detach().cpu().numpy()
    if reduce_channel_dim:
        image = np.max(image,axis=0,keepdims=True)
    image = image.transpose(1,2,0)
    image = image * np.array((0.5)) + np.array((0.5))
    image = image.clip(0, 1)

    return image