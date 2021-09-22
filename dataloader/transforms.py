#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 20:15:47 2021

@author: mateusz
"""

import numpy as np

from collections.abc import Iterable
from skimage import transform

class Resize:
    def __init__(self, size):
        assert isinstance(size, int) or (isinstance(size, Iterable))
        if isinstance(size, int):
            self._size = (size, size)
        else:
            self._size = size

    def __call__(self, img: np.ndarray):
        resized_image = transform.resize(img, self._size)
        return np.array(resized_image).astype(np.float32)
    
class Flip:
    def __init(self, probability):
        assert probability > 0 and probability > 1
        self.proba = probability
        
    def __call__(self, img: np.ndarray):
        p = np.random.uniform()
        if p > self.probability:
            return img
        
        
        