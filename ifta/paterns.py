# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 08:45:19 2024

@author: f24lerou
"""

import numpy as np
import matplotlib.pyplot as plt

def cross(sizeCross, **kargs):

#8<---------------------------------------------------------------------------------------------
# generateCircularPupil : generate a circular pupil over a square support
#
# Author : Francois Leroux
# Contact : francois.leroux.pro@gmail.com
# Status : in progress
# Last update : 2024.02.28, Brest
# Comments : for even support size, coordinates are defined like [-2,-1,0,1] (N = 4)
#
# Inputs : MANDATORY : sizeCross {integer}[pixel]
#
#          OPTIONAL :  sizeSupport : resolution of the support {tupple (1x2)}[pixel] - default value : sizeCross
#                      center : cartesian coordinates {tupple (1x2)}[pixel] - default value : [0,0]
#                      width : width of the cross {integer}[pixel] - default value : 1
#                     
# Outputs : a binary cross
#8<---------------------------------------------------------------------------------------------

    # read optinal parameters values
    sizeSupport = kargs.get("sizeSupport", [sizeCross,sizeCross])
    center = kargs.get("center", [0,0])
    width = kargs.get("width", 1)
        
    cross = np.zeros(sizeSupport)
    
    cross[sizeSupport[0]//2-center[0]-width//2:sizeSupport[0]//2-center[0]+width//2+width%2, 
          sizeSupport[1]//2+center[1]-sizeCross//2:sizeSupport[1]//2+center[1]+sizeCross//2+sizeCross%2] = 1
    
    cross[sizeSupport[0]//2-center[0]-sizeCross//2:sizeSupport[0]//2-center[0]+sizeCross//2+sizeCross%2, 
          sizeSupport[1]//2+center[1]-width//2:sizeSupport[1]//2+center[1]+width//2+width%2] = 1
    
    return cross

# sizeCross = 11
# sizeSupport = [32,32]

# dirc = r"D:\francoisLeroux\codes\ifta\patern\outputs\\"
# filename = "cross_"+str(sizeCross)+"_"+str(sizeSupport[0])+"x"+str(sizeSupport[1])+".npy"

# cross = cross(sizeCross, sizeSupport = sizeSupport, width=5)

# np.save(dirc+filename, cross)

# cross = np.load(dirc+filename)

# plt.imshow(cross)

