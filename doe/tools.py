# -*- coding: utf-8 -*-
"""
Created on Thu Feb 29 13:44:30 2024 under Python 3.11.7

@author: f24lerou
"""

import numpy as np

def computeFocal(d1, d2):
    
#8<---------------------------------------------------------------------------------------------
# computeFocal : compute the focal lenght needed for conjugating a point at distance d1 with respect to the lens in object space
#                to a point at distance d2 with respect to the lens in image space
#                Descartes formula: 1/d2 + 1/d1 = 1/f => f = d1*d2/(d1+d2)
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

def discretization(phase, n_levels):
    
    #8<---------------------------------------------------------------------------------------------
    # discretization : return an array of phase values between 0 and 2pi, discretized over n_levels
    #                               
    # Author : Francois Leroux
    # Contact : francois.leroux.pro@gmail.com
    # Status : done
    # Last update : 2024.03.01, Brest
    #
    # Comments :
    #
    # Inputs : MANDATORY : phase {float}
    #                      n_levels {int} 
    #                     
    # Outputs : phase : the discretized phase, values between -pi and pi
    #8<---------------------------------------------------------------------------------------------
    
    phase = np.angle(np.exp(1j*phase))                                # phase values between -pi and pi 
    
    if n_levels == 0:
        
        return phase
    
    else:
    
        #phase = np.angle(np.exp(1j*phase))                            # phase values between -pi and pi 
        phase = np.round((phase+np.pi)/ (2*np.pi) * (n_levels-1))     # phase discretization. phase values between 0 and n_levels-1
        phase = 2*np.pi / (n_levels-1) * phase - np.pi                # discretized phase between -pi and pi
    
    return phase