# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 11:16:02 2024

@author: f24lerou
"""

import numpy as np

def lens(f, *, sizeSupport=[128, 128], sideLength=0.02):
    
#8<---------------------------------------------------------------------------------------------
# lens : generate a phase screen correspnding to a thin lens under paraxial approximation
#
# Author : Francois Leroux
# Contact : francois.leroux.pro@gmail.com
# Status : in progress
# Last update : 2024.02.28, Brest
# Comments : For even support size, coordinates are defined like [-2,-1,0,1] (N = 4)
#            Source of the formula: Introduction to Fourier Optics, J.W Goodman, p.99
#            Convention: exp(-j omega t) ??
#   
# Inputs : MANDATORY : f : focal length of the lens {float}[m] >0 => convergent of divergent lens ??
#
#          OPTIONAL :  sizeSupport : resolution of the support {tupple (1x2)}[pixel] - default value: [128, 128]
#                      sizeLength : physical length of the side of the support {float}[m] - default value: 2 cm
#                     
# Outputs : pm: phase mask
#8<---------------------------------------------------------------------------------------------
    
    [X, Y] = np.meshgrid(np.arange(-sizeSupport[1]//2+sizeSupport[1]%2, sizeSupport[1]//2+sizeSupport[1]%2), 
                         np.arange(sizeSupport[0]//2, -sizeSupport[0]//2, step=-1))
    
    return [X, Y]
    