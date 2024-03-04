# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 08:45:19 2024 under Python 3.11.7

@author: f24lerou
"""

import numpy as np

def cross(sizeCross, *, center=[0,0], width=1,**kargs):

#8<---------------------------------------------------------------------------------------------
# cross : generate a cross over a square or rectangular support
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
        
    cross = np.zeros(sizeSupport)
    
    cross[sizeSupport[0]//2-center[0]-width//2:sizeSupport[0]//2-center[0]+width//2+width%2, 
          sizeSupport[1]//2+center[1]-sizeCross//2:sizeSupport[1]//2+center[1]+sizeCross//2+sizeCross%2] = 1
    
    cross[sizeSupport[0]//2-center[0]-sizeCross//2:sizeSupport[0]//2-center[0]+sizeCross//2+sizeCross%2, 
          sizeSupport[1]//2+center[1]-width//2:sizeSupport[1]//2+center[1]+width//2+width%2] = 1
    
    return cross
