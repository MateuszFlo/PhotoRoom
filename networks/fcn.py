#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 19:28:36 2021

@author: mateusz
"""
import torch.nn as nn
from networks.utils import initialize_weights, activation_fct

class FCN2D(nn.Module):
    def __init__(self, input_channels=3, hidden_channels=32, 
                 output_channels=15, layers=3, filter_size=(3,3), 
                 padding=(1,1), padding_mode='zeros', bias=None, 
                 desired_activation='relu', normalization=nn.InstanceNorm2d):
        super().__init__()
        activation = activation_fct(desired_activation)
        
        layerlist = [nn.Conv2d(input_channels, hidden_channels, 
                               kernel_size=(1,1)),
                     activation]

        for l in range(layers):
            layerlist += [nn.Conv2d(hidden_channels, hidden_channels, 
                                    kernel_size=filter_size, padding=padding, 
                                    padding_mode=padding_mode, bias=bias)]
            if normalization:
                layerlist += [normalization(hidden_channels)]
            layerlist += [activation]
                                                 
        layerlist += [nn.Conv2d(hidden_channels, output_channels, kernel_size=(1,1))]  

        
        self.net = nn.Sequential(*layerlist)
        self.net.apply(initialize_weights)
            
    def forward(self, x):
        return self.net(x)

class Conv2DBlock(nn.Module):
    def __init__(self, input_channels, hidden_channels, 
                 kernel_size=(3,3), padding=(1,1), padding_mode='zeros', 
                 bias=None, desired_activation='relu', 
                 normalization=nn.InstanceNorm2d, dropout=None):
        super().__init__()
        activation = activation_fct(desired_activation)
        
        layers = [nn.Conv2d(input_channels, hidden_channels, 
                            kernel_size, padding=padding, 
                            padding_mode=padding_mode, bias=bias)]
        if normalization:
            layers += [normalization(hidden_channels)]
        layers += [activation]
        if dropout:
            layers += [nn.Dropout2d(p=dropout)]
            
        layers += [nn.Conv2d(hidden_channels, hidden_channels, 
                             kernel_size, padding=padding, 
                             padding_mode=padding_mode, bias=bias)]
        if normalization:
            layers += [normalization(hidden_channels)]
        layers += [activation]
        if dropout:
            layers += [nn.Dropout2d(p=dropout)]

            
        self.model = nn.Sequential(*layers)
        self.model.apply(initialize_weights)
        
    def forward(self, x):
        x = self.model(x)
        return x

class Pooled_FCN2D(nn.Module):
    def __init__(self, input_channels=3, hidden_channels=32, conv=Conv2DBlock,
                 output_channels=15, layers=3, filter_size=(3,3), 
                 padding=(1,1), padding_mode='zeros', bias=None, 
                 desired_activation='relu', normalization=nn.InstanceNorm2d):
        super().__init__()
        down = nn.MaxPool2d((2,2))
        up = nn.Upsample(scale_factor=2)
        activation = activation_fct(desired_activation)
        
        layerlist = [nn.Conv2d(input_channels, hidden_channels, 
                               kernel_size=(1,1)),
                     activation]

        for l in range(layers):
            layerlist += [conv(hidden_channels*(2**l), 
                               hidden_channels*(2**(l+1)), 
                               kernel_size=filter_size, padding=padding, 
                               padding_mode=padding_mode, bias=bias)]
            if normalization:
                layerlist += [normalization(hidden_channels*(2**(l+1)))]
            layerlist += [activation]
            layerlist += [down]
            
        
        
        for l in reversed(range(layers)):
            layerlist += [up,
                          conv(hidden_channels*(2**(l+1)), 
                                hidden_channels*(2**l), 
                                kernel_size=filter_size, padding=padding, 
                                padding_mode=padding_mode, bias=bias)]
            if normalization:
                layerlist += [normalization(hidden_channels)]
            layerlist += [activation]
                                                  
        layerlist += [nn.Conv2d(hidden_channels, output_channels, kernel_size=(1,1))]  

        
        self.net = nn.Sequential(*layerlist)
        self.net.apply(initialize_weights)
            
    def forward(self, x):
        return self.net(x)
    

    
if __name__ == "__main__":
    import torch
    dummy = torch.randn(1,3,96,96)
    net = FCN2D()
    out = net(dummy)
    assert(dummy.shape[2:] == out.shape[2:])
    net = Pooled_FCN2D()
    out = net(dummy)
    assert(dummy.shape[2:] == out.shape[2:])