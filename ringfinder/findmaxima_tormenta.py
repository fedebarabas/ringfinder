# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 15:43:20 2016

@author: Cibion
"""



import numpy as np
from PIL import Image
import maxima as mx

im = Image.open('spectrin1.tif')
data = np.array(im)

maxObject = mx.Maxima(data)
maxObject.find()
maxObject.positions

