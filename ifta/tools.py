# -*- coding: utf-8 -*-
"""
Created on Thu Feb 29 13:44:30 2024

@author: f24lerou
"""

def computeFocal(d1, d2):
    
#8<---------------------------------------------------------------------------------------------
# computeFocal : compute the focal lenght needed for conjugating a point at distance d1 with respect to the lens in object space
#                to a point at distance d2 with respect to the lens in image space
#                               
# Author : Francois Leroux
# Contact : francois.leroux.pro@gmail.com
# Status : in progress
# Last update : 2024.02.29, Brest
#
# Comments :
#
# Inputs : MANDATORY : d1 {float}[m]
#                      d2 {float}[m] 
#                     
# Outputs : f : the focal lenght of the corresponding convergent lens
#8<---------------------------------------------------------------------------------------------

    f = d1*d2/(d1+d2)
    
    return f