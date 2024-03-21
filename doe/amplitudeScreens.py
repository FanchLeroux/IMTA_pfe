# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 14:54:58 2024     under Python 3.11.7

@author: f24lerou
"""

# 8<---------------------------- Add path --------------------------------

import os
import sys

path = os.path.abspath(os.path.abspath('..'))
sys.path.append(path)

# 8<--------------------------- Import modules ---------------------------

import numpy as np

from tools import getCartesianCoordinates 

# 8<------------------------- Functions definitions ----------------------


def Gaussian(size_support, pixel_pitch, sigma):
    """
    divergenceToWaist :  compute the waist from the divergence
                      
    Author : Francois Leroux
    Contact : francois.leroux.pro@gmail.com
    Status : in progress
    Last update : 2024.03.18, Brest
    Comments : sigma = half width at 1/e / sqrt(2)
    
    Inputs : MANDATORY :  wavelength {float}[m] : wavelength
                          divergence {float}[°] : the laser divergence (full angle)
                                 
    Outputs : w_0 {float}[m] : object waist
    """
    [X,Y] = getCartesianCoordinates(size_support)*pixel_pitch
    gaussain_amplitude = np.exp(-(X**2+Y**2)/(2*sigma**2))
    
    return gaussain_amplitude