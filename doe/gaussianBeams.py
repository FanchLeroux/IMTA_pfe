# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 14:23:09 2024    under Python 3.11.7

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

import sympy

# 8<------------------------- Functions definitions ----------------------

def getGaussianBeamRadius(wavelength, divergence, propagation_distance):
    
#8<---------------------------------------------------------------------------------------------
# getGaussianBeamDiameter : compute the radius (half width at 1/e of the maximum amplitude, i.e 1/e^2 of the maximum intensity)
#                           for a gaussian beam with a given divergence that has been propagated over a given distance 
# Author : Francois Leroux
# Contact : francois.leroux.pro@gmail.com
# Status : in progress
# Last update : 2024.03.08, Brest
# Comments : gaussian beam propagation : https://fr.wikipedia.org/wiki/Faisceau_gaussien 
#   
# Inputs : MANDATORY : wavelength [m] : the laser wavelength
#                      propagation_distance [m] : the propagation distance  
#                      divergence {float}[°] : the laser divergence (full angle)
#                    
# Outputs : w_z : the half width at 1/e of the maximum amplitude
#8<-----------------------------------------------------------------------------------------------
    
    w_0 = wavelength/(np.pi * np.tan(np.pi/180 * divergence/2))        
    z_0 = np.pi*w_0**2/wavelength # Rayleigh length
    w_z = w_0 * (1 + (propagation_distance / z_0)**2)**0.5 # half width at 1/e of the maximum amplitude
    
    return w_z

def getMinimalCollectorLength(w_z, efficiency):
    
#8<---------------------------------------------------------------------------------------------
# getMinimalCollectorLength : compute minimal length
# Author : Francois Leroux
# Contact : francois.leroux.pro@gmail.com
# Status : in progress
# Last update : 2024.03.08, Brest
# Comments : gaussian beam propagation : https://fr.wikipedia.org/wiki/Faisceau_gaussien 
#   
# Inputs : MANDATORY : w_z : the half width at 1/e of the maximum amplitude
#                      efficiency : ratio energy collected / energy emmited
#                    
# Outputs : length_mini : get the side length of a square collector that collect efficiency * energy emmited
#8<-----------------------------------------------------------------------------------------------    
    
    x = sympy.Symbol("x")
    y = sympy.Symbol("y")
    irradiance = sympy.exp(-(x**2+y**2)/w_z**2)**2
    x_half_extent = sympy.Symbol("x_half_extent")
    x_half_extent_mini = sympy.solvers.solve(sympy.integrate(irradiance, (x, -x_half_extent, x_half_extent), (y, -x_half_extent, x_half_extent))
                               /(np.pi * w_z**2 / 2)-efficiency, x_half_extent)
    length_mini = 2*x_half_extent_mini[1] # retain only positive value
        
    return float(length_mini)

def getFocalLength(d1, d2, wavelength, divergence):

#8<---------------------------------------------------------------------------------------------
# getFocalLength : compute the focal length of a thin lens in order to conjugate the object waist of a gaussian beam at a 
#                  distance d1 from the lens to the image waist at a distance d2 from the lens, according to the modified 
#                  thin lens equation.  
#                  
# Author : Francois Leroux
# Contact : francois.leroux.pro@gmail.com
# Status : in progress
# Last update : 2024.03.11, Brest
# Comments : modified thin lens equation : https://www.thorlabs.com/newgrouppage9.cfm?objectgroup_id=14511 
#            WARNING : can return negative values   
#
# Inputs : MANDATORY : d1 {float}[m] : absolute distance object point - lens
#                      d2 {float}[m] : absolute distance lens - image point 
#                      wavelength {float}[m] : wavelength
#                      divergence {float}[°] : the laser divergence (full angle)
#                     
# Outputs : f_modified_thin_lens_formula : the focal lenght of the corresponding convergent lens
#8<-----------------------------------------------------------------------------------------------        
    
    w_0 = wavelength/(np.pi * np.tan(np.pi/180 * divergence/2))
    zr = np.pi * w_0**2 / wavelength # object space Rayleigh range
    f = sympy.Symbol("f")
    f_modified_thin_lens_formula = sympy.solvers.solve(1/(d1+zr**2/(d1-f))+1/d2-1/f, f)
    f_modified_thin_lens_formula = [float(f_modified_thin_lens_formula[0]), float(f_modified_thin_lens_formula[1])]
    
    f_thin_lens_formula = d1*d2/(d1+d2)
    
    diff = [float(f_modified_thin_lens_formula[0])-f_thin_lens_formula, 
            float(f_modified_thin_lens_formula[1])-f_thin_lens_formula]
    
    f_modified_thin_lens_formula = f_modified_thin_lens_formula[diff==min(diff)]
    diff = min(diff)
    
    return f_modified_thin_lens_formula, diff


def gaussianEfficiency(wavelength, distance, x_half_extent, **kargs):
    
#8<---------------------------------------------------------------------------------------------
# gaussianEfficiency : (Unused) compute the efficiency, i.e the ratio between the light emmited and collected 
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
    divergence = kargs.get("divergence", np.nan)
    w_0 = kargs.get("w_0", wavelength/(np.pi * np.tan(np.pi/180 * divergence/2)))
    divergence = kargs.get("divergence", 2 * np.arctan(wavelength / (np.pi*w_0)) * 180/np.pi) # [deg]
    
    divergence = np.pi/180 * divergence # [deg] to [rad] conversion
    
    z_0 = np.pi*w_0**2/wavelength # Rayleigh length
    w_z = w_0 * (1 + (distance / z_0)**2)**0.5 # half width at 1/e of the maximum amplitude
    
    f = lambda y, x: np.exp(-(x**2+y**2)/w_z**2)**2 # irradince = amplitude^2 
    
    energy, _ = integrate.dblquad(f, -x_half_extent, x_half_extent, -y_half_extent, y_half_extent)
    energy_total = np.pi * w_z**2 / 2 # Gauss integral, irradiance
        
    return energy/energy_total