# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 15:50:46 2024 under Python 3.11.7

@author: f24lerou

Purpose : set of functions allowing to run an entire hologram design, with or without replication

Comments : Vocabulary : "holo" often refers to one period of the replicated hologram
                        "optic" often refers to the entire optical component being designed,
                        i.e the replicated hologram plus the fresnel lens

"""

# 8<------------------------------------------ Add path ---------------------------------------

import os
import sys

path = os.path.abspath(os.path.abspath('..')) # path = 'D:\\francoisLeroux\\codes'
sys.path.append(path)

#%% 8<-------------------------------------- Import modules -----------------------------------

import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid1 import make_axes_locatable

from doe.paterns import cross, gridSquares

from doe.phaseScreens import lens, getOpticSideLengthMaxi
from doe.amplitudeScreens import Gaussian

from doe.tools import discretization, getCartesianCoordinates, ZerosPadding
from doe.gaussianBeams import getGaussianBeamRadius, getCollectorLengthMini, getFocalLength, \
                              gaussianEfficiency, getImageWaist, divergenceToWaist
from doe.ifta import ifta, iftaSoftQuantization





def GetOpticLengthMinMax(wavelength, divergence, d1, d2, light_collection_efficiency_mini, fringe_length_mini):
    
    """
    GetOpticLengthMinMax : return the maximal and minimal side length of the optic given
                           a minimal value for the ratio between the light collected and 
                           emmited and a minimal value for the width of the fringes of the 
                           Fresnel lens that can be fabricated 
                                  
    Author : Francois Leroux
    Contact : francois.leroux.pro@gmail.com
    Status : in progress
    Last update : 2024.03.24, Brest
    
    Comments : n_levels should have an impact on getOpticSideLengthMaxi for this function 
               to work with n_levels != 2
    
    Inputs : MANDATORY : wavelength {float}[m]
                         divergence {float}[°] : gaussian beam divergence, i.e full angle at 1/e of the max amplitude
                         d1 {float}[m] : absolute distance object point - lens
                         d2 {float}[m] : absolute distance lens - image point
                         light_collection_efficiency_mini {float} : minimal ratio between the energy emitted by the VCSEL and 
                             the incident energy on the hologram. Default value : 0.5
                         fringe_length_mini {float}[m] : fabrication constaint : minimal width of the fringes at the
                             edges of the fresnel lens (half a period)
           
                        
                                                 
                        
    Outputs : optic_length_mini, optic_length_maxi
    """
    
    # Fresnel lens focal length
    [f, diff] = getFocalLength(d1, d2, wavelength, divergence) # focal length for source - image plane conjugation
    
    # optic side length mini to match light collection requirement
    w_z = getGaussianBeamRadius(wavelength=wavelength, divergence=divergence, propagation_distance=d1)
    optic_length_mini = getCollectorLengthMini(w_z=w_z, efficiency=light_collection_efficiency_mini)

    # optic side length maxi to match thin fringes requirement
    optic_length_maxi = getOpticSideLengthMaxi(wavelength, f, fringe_length_mini)
    
    return optic_length_mini, optic_length_maxi, f



def GetHoloSize(wavelength, d2, separation, optic_pp, optic_length):

    """
    GetHoloSize : return the size in pixel of  
                                  
    Author : Francois Leroux
    Contact : francois.leroux.pro@gmail.com
    Status : in progress
    Last update : 2024.03.24, Brest
    
    Comments : 
    
    Inputs : MANDATORY : wavelength {float}[m]
                         d2 {float}[m] : absolute distance lens - image point
                         separation {float}[m] : separation between two image samples 
                             dots step of the Dirac comb in image plane (= target_pp)
                         optic_pp {float}[m] : pixel pitch in hologram plane, depends on
                             the photoplotteur capabilities
                         optic_length {float}[m] : desired optic length (will be modified to
                             match the number of replications                       
                        
    Outputs : 
    """

    holo_length = wavelength * d2 / separation  # [m] step of the Dirac comb in optic space, i.e separation
                                                # between two replicated holograms, i.e hologram side length
    
    holo_size = int(holo_length//optic_pp - (holo_length//optic_pp)%2)
    
    holo_length = holo_size * optic_pp
    
    separation = wavelength * d2 / holo_length
    
    n_replications = int(optic_length//holo_length)
    optic_length = n_replications * holo_length # [m] final optic side length                                            
    
    return holo_size, n_replications, optic_length, separation
                                                
                                                
                                                
                                                
                                                
                                                
                                                
                                                
                                                
                                                
                                                
                                                
                                                
                                                
                                                

