# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 08:45:19 2024 under Python 3.11.7

@author: f24lerou
"""

# 8<---------------------------- Add path --------------------------------

import os
import sys

path = os.path.abspath(os.path.abspath('..'))
sys.path.append(path)

# 8<--------------------------- Import modules ---------------------------

import numpy as np
from doe.tools import getCartesianCoordinates

def cross(cross_size, *, center=[0,0], width=1,**kargs):

#8<---------------------------------------------------------------------------------------------
# cross : generate a cross over a square or rectangular support
#
# Author : Francois Leroux
# Contact : francois.leroux.pro@gmail.com
# Status : in progress
# Last update : 2024.02.28, Brest
# Comments : for even support size, coordinates are defined like [-2,-1,0,1] (N = 4)
#
# Inputs : MANDATORY : cross_size {integer}[pixel]
#
#          OPTIONAL :  support_size : resolution of the support {tupple (1x2)}[pixel] - default value : cross_size
#                      center : cartesian coordinates {tupple (1x2)}[pixel] - default value : [0,0]
#                      width : width of the cross {integer}[pixel] - default value : 1
#                     
# Outputs : a binary cross
#8<---------------------------------------------------------------------------------------------

    # read optinal parameters values
    support_size = kargs.get("support_size", [cross_size,cross_size])
        
    cross = np.zeros(support_size)
    
    cross[support_size[0]//2-center[0]-width//2:support_size[0]//2-center[0]+width//2+width%2, 
          support_size[1]//2+center[1]-cross_size//2:support_size[1]//2+center[1]+cross_size//2+cross_size%2] = 1
    
    cross[support_size[0]//2-center[0]-cross_size//2:support_size[0]//2-center[0]+cross_size//2+cross_size%2, 
          support_size[1]//2+center[1]-width//2:support_size[1]//2+center[1]+width//2+width%2] = 1
    
    return cross

def gridSquares(nPointsX, *, spacing=1, width=1, margin=0, **kargs):
    
#8<---------------------------------------------------------------------------------------------
# gridSquares : generate a grid of squares
#
# Author : Francois Leroux
# Contact : francois.leroux.pro@gmail.com
# Status : in progress
# Last update : 2024.03.06, Brest
# Comments : for even support size, coordinates are defined like [-2,-1,0,1] (N = 4)
#
# Inputs : MANDATORY : nPointsX {int} : number of squares in the X direction
#
#          OPTIONAL :  nPointsY {int} : number of squares in the Y direction
#                      spacing {int}[pixel] : space between two adjacent squares in the x or y direction - default value : 1
#                      width : side length of the squares {integer}[pixel] - default value : 1
#                      margin {int}[pixel] : space between the edges of the array and the firsts and lasts squares
#                     
# Outputs : a grid of squares
#8<---------------------------------------------------------------------------------------------    
        
    # read optinal parameters values
    nPointsY = kargs.get("nPointsY", nPointsX)
    
    point = np.ones([width, width])
    
    grid = np.zeros([nPointsY*width + (nPointsY-1)*spacing+2*margin, nPointsX*width + (nPointsX-1)*spacing+2*margin])
    
    for i in range(nPointsY):
        for j in range (nPointsX):
            grid[margin+i*(width+spacing):margin+i*(width+spacing)+width, 
                       margin+j*(width+spacing):margin+j*(width+spacing)+width] = point
            
    return grid
            
            
            
            
    
    
    
    
    
    
    
    
    
    
    
    