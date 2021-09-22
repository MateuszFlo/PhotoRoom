#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 19:40:46 2021

@author: mateusz
"""
import math
import torch
import torch.nn as nn


def initialize_weights(layer):
    if type(layer) == nn.Conv2d:
        he_normal(layer.weight)
    if type(layer) == nn.ConvTranspose2d:
        he_normal(layer.weight)
    if type(layer) == nn.Linear:
        he_normal(layer.weight)
        
def he_normal(tensor, gain=1):
    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(2.0 / (fan_in))
    with torch.no_grad():
        return tensor.normal_(0, std)
    
def activation_fct(activation):
    return  nn.ModuleDict([
        ['relu', nn.ReLU(inplace=True)],
        ['leaky_relu', nn.LeakyReLU(negative_slope=0.01, inplace=True)],
        ['selu', nn.SELU(inplace=True)],
        ['none', nn.Identity()]
    ])[activation]