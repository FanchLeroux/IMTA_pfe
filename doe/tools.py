# -*- coding: utf-8 -*-
"""
Created on Thu Feb 29 13:44:30 2024 under Python 3.11.7

@author: f24lerou
"""

# 8<---------------------------- Add path --------------------------------

import os
import sys

path = os.path.abspath(os.path.abspath('..'))
sys.path.append(path)

# 8<--------------------------- Import modules ---------------------------

import numpy as np
from scipy import integrate

# 8<------------------------- Functions definitions ----------------------

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


def getCartesianCoordinates(nrows, **kargs):

#8<---------------------------------------------------------------------------------------------
# getCartesianCoordinates : generate two arrays representing the cartesian coordinates
#
# Author : Francois Leroux
# Contact : francois.leroux.pro@gmail.com
# Status : in progress
# Last update : 2024.03.05, Brest
# Comments : For even support size, coordinates are defined like [-2,-1,0,1] (N = 4)
#   
# Inputs : MANDATORY : nrows {int} : the number of rows
#
#          OPTIONAL : ncols {int} : the number of columns
#                    
# Outputs : phase, values between -pi and pi
#8<---------------------------------------------------------------------------------------------
    
    # read optinal parameters values
    ncols = kargs.get("ncols", nrows)

    [X, Y] = np.meshgrid(np.arange(-ncols//2+ncols%2, ncols//2+ncols%2), 
                         np.arange(nrows//2-1+nrows%2, -nrows//2-1+nrows%2, step=-1))
    
    return [X,Y]


def gaussianEfficiency(wavelength, distance, x_half_extent, **kargs):
    
#8<---------------------------------------------------------------------------------------------
# gaussianEfficiency : compute the efficiency, i.e the ratio between the light emmited and collected 
#                      in the transverse plane for a given extent
#
# Author : Francois Leroux
# Contact : francois.leroux.pro@gmail.com
# Status : in progress
# Last update : 2024.03.05, Brest
# Comments : integration methods:        https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.dblquad.html#scipy.integrate.dblquad
#                                        https://fr.wikipedia.org/wiki/Int%C3%A9grale_de_Gauss
#            
#            gaussian beam propagation : https://fr.wikipedia.org/wiki/Faisceau_gaussien 
#   
# Inputs : MANDATORY : wavelength [m] : the laser wavelength
#                      distance [m] : the propagation distance 
#                      at least one optional keyword argument, either divergence or w_0 
#
#          OPTIONAL : w_0 [m] : the laser waist
#                     divergence {float}[°] : the laser divergence (full angle)  
#                    
# Outputs : efficiency : the efficiency
#8<---------------------------------------------------------------------------------------------    
    
    y_half_extent = kargs.get("y_half_extent", x_half_extent)
    divergence = kargs.get("divergence", None)
    w_0 = kargs.get("w_0", wavelength/(np.pi * np.tan(np.pi/180 * divergence/2)))
    divergence = kargs.get("divergence", 2 * wavelength / (np.pi*w_0) * 180/np.pi) # [deg]
    
    divergence = np.pi/180 * divergence # [deg] to [rad] conversion
    
    z_0 = np.pi*w_0**2/wavelength
    w_z = w_0 * (1 + (distance / z_0)**2)**0.5 # half width at 1/e of the maximum amplitude
    
    f = lambda y, x: np.exp(-(x**2+y**2)/w_z**2)**2 # irradince = amplitude^2 
    
    energy, _ = integrate.dblquad(f, -x_half_extent, x_half_extent, -y_half_extent, y_half_extent)
    energy_total = np.pi * w_z**2 / 2 # Gauss integral, irradiance
        
    return energy/energy_total, w_z, w_0




