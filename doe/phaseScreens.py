# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 11:16:02 2024 under Python 3.11.7

@author: f24lerou
"""

# 8<---------------------------- Add path --------------------------------

import os
import sys

path = os.path.abspath(os.path.abspath('..'))
sys.path.append(path)

# 8<--------------------------- Import modules ---------------------------

import numpy as np
import sympy
from doe.tools import discretization

# 8<------------------------- Functions definitions ----------------------


def lens(f, *, wavelength=0.5e-6, sizeSupport=[128, 128], samplingStep=1e-4, n_levels=0):
    
#8<---------------------------------------------------------------------------------------------
# lens : generate a phase screen correspnding to a thin lens under paraxial approximation
#
# Author : Francois Leroux
# Contact : francois.leroux.pro@gmail.com
# Status : in progress
# Last update : 2024.02.28, Brest
# Comments : For even support size, coordinates are defined like [-2,-1,0,1] (N = 4)
#            Source of the formula: Introduction to Fourier Optics, J.W Goodman, p.99
#   
# Inputs : MANDATORY : f : focal length of the lens {float}[m], f>0 => convergent lens, f<0 => divergent lens
#
#          OPTIONAL :  wavelenght : wavelenght {float}[m] - default value: 0.5 µm
#                      sizeSupport : resolution of the support {tupple (1x2)}[pixel] - default value: [128, 128]
#                      samplingStep : physical length of on pixel of the support {float}[m] - default value: 1 mm
#                      n_levels : number of levels over which the phase needs to be quantified. {int} - default value: 0, no discretization
#                     
# Outputs : phase, values between -pi and pi
#8<---------------------------------------------------------------------------------------------
    
    [X, Y] = np.meshgrid(np.arange(-sizeSupport[1]//2+sizeSupport[1]%2, sizeSupport[1]//2+sizeSupport[1]%2), 
                         np.arange(sizeSupport[0]//2, -sizeSupport[0]//2, step=-1))
    
    X = samplingStep * X
    Y = samplingStep * Y
    
    phase = 2*np.pi/wavelength * 1/(2*f) * (X**2 + Y**2)
    
    if n_levels != 0:
         phase = discretization(phase, n_levels)
    
    return phase
    

def tilt(deltaPhi, *, sizeSupport=[128, 128], samplingStep=1e-4, n_levels=0):
    
#8<---------------------------------------------------------------------------------------------
# tilt : generate a phase screen correspnding to a tilt in x
#
# Author : Francois Leroux
# Contact : francois.leroux.pro@gmail.com
# Status : in progress
# Last update : 2024.03.04, Brest
# Comments : For even support size, coordinates are defined like [-2,-1,0,1] (N = 4)
#   
# Inputs : MANDATORY : deltaPhi [rad] : absolute phase difference between the edges of the phase screen
#
#          OPTIONAL :  wavelenght : wavelenght {float}[m] - default value: 0.5 µm
#                      sizeSupport : resolution of the support {tupple (1x2)}[pixel] - default value: [128, 128]
#                      n_levels : number of levels over which the phase needs to be quantified. {int} - default value: 0, no discretization
#                     
# Outputs : phase, values between -pi and pi
#8<---------------------------------------------------------------------------------------------

    [X, Y] = np.meshgrid(np.arange(0,sizeSupport[0]), np.arange(0,sizeSupport[1]))
    
    X = np.asarray(X, dtype=np.float32)
    
    X /= np.float32(sizeSupport[0])
    
    phase = X * deltaPhi
    
    phase = discretization(phase, n_levels)
    
    return phase

def getOpticSideLengthMaxi(wavelength, f, fringe_length_mini):
    
#8<---------------------------------------------------------------------------------------------
# getOpticSideLengthMaxi : compute the maximal side length of a binary Fresnel lens phase screen 
#                          in order to avoid too thin fringes at the edges
#
# Author : Francois Leroux
# Contact : francois.leroux.pro@gmail.com
# Status : in progress
# Last update : 2024.03.11, Brest
# Comments : Delta_R (radial distance over which we have a 2pi phase shift) = (-2R + sqrt(4R^2+8*wavelength*f))/2
#            with R the radial coordinate                                     
#   
# Inputs : MANDATORY : wavelength {float}[m] : wavelength
#                      f {float}[m] : focal length of the lens
#                      fringe_length_mini {float}[m] : minimal length for a fringe (1 pseudo periode = 2 fringes)
#                      pixel_pitch {float}[m] : pixel pitch
#                     
# Outputs : side_length_maxi {float} [m] : maximal side length of a binary Fresnel lens phase screen 
#                                          in order to avoid too thin fringes at the edges
#8<---------------------------------------------------------------------------------------------    
    
    side_length_maxi = sympy.Symbol("side_length_maxi")
    side_length_maxi = np.asarray(sympy.solvers.solve(((4*side_length_maxi**2+8*wavelength*f)**0.5-2*side_length_maxi)/2-2*fringe_length_mini, 
                        side_length_maxi), dtype = float)
    side_length_maxi = 2*side_length_maxi[side_length_maxi>=0]/2**0.5

    return side_length_maxi[0]




