# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 09:58:42 2024 under Python 3.11.7

@author: f24lerou
"""

# 8<---------------------------- Add path --------------------------------

import os
import sys

path = os.path.abspath(os.path.abspath('..'))
sys.path.append(path)

# 8<--------------------------- Import modules ---------------------------

import numpy as np
from doe.tools import discretization

# 8<------------------------- Functions definitions ----------------------



def ifta(target, doe_size, *, n_iter = 25, rfact = 1.2, n_levels = 0, compute_efficiency = 0):

#8<---------------------------------------------------------------------------------------------
# Iterative Fourier Transform Algorithm
#
# Author : Francois Leroux
# Contact : francois.leroux.pro@gmail.com
# Status : in progress
# Last update : 2024.03.01, Brest
# Comments : for even support size, coordinates are defined like [-2,-1,0,1] (N = 4)
#
# Inputs : MANDATORY : target : image we want to get at infinity under plane wave illumination  {2D float np.array}[Irradiance]
#                      doe_size : size of the DOE {tupple (1x2)}[pixel] - should have larger elements than target_image.shape 
#
#          OPTIONAL :  n_iter : number of iteration of each loop of the algorithm {int} - default value = 25
#                      r_fact : reinforcment factor. Forces the energy to stay in the ROI - default value = 1.2
#                      n_levels : number of levels over which the phase will be discretized - default value = 0 : no discretization
#                      compute_efficiency : {bool} If 1, efficiency is computed and returned - default value = 0
#                     
# Outputs : a binary cross
#8<---------------------------------------------------------------------------------------------
    
    target_size = target.shape
    
    target_amp = np.asarray(target, float)       # conversion target to float
    target_amp = np.sqrt(target_amp)             # get target amplitude

    amp_image = np.zeros(doe_size)                                                                             # Amplitude output field = 0
    amp_image[doe_size[0]//2-target_size[0]//2:doe_size[0]//2-target_size[0]//2+target_size[0], 
              doe_size[1]//2-target_size[1]//2:doe_size[1]//2-target_size[1]//2+target_size[1]] = target_amp   # Amplitude = target image in window
    phase_image = 2*np.pi*np.random.rand(doe_size[0], doe_size[1])                                             # Random image phase    
    field_image = amp_image*np.exp(1j * phase_image)                                                           # Initiate input field
    
    # First loop - continous phase screen computation
    
    for iter in range(n_iter):
        field_DOE = np.fft.ifft2(np.fft.ifftshift(field_image))                                                         # field DOE = TF-1 field image
        phase_DOE = np.angle(field_DOE)                                                                                 # save DOE phase
        field_DOE = np.exp(phase_DOE * 1j)                                                                              # force the amplitude of the DOE to 1 (no losses)
        field_image = np.fft.fftshift(np.fft.fft2(field_DOE))                                                           # field image = TF field DOE
        phase_image = np.angle(field_image)                                                                             # save image phase
        amp_image[doe_size[0]//2-target_size[0]//2:doe_size[0]//2-target_size[0]//2+target_size[0], 
                  doe_size[1]//2-target_size[1]//2:doe_size[1]//2-target_size[1]//2+target_size[1]] = rfact*target_amp  # force the amplitude of the DOE to the target amplitude inside the ROI. Outside the ROI, the amplitude is free
        field_image = amp_image*np.exp(phase_image * 1j)       # new image field computation

    # Second loop - discretized phase screen

    if n_levels != 0:

        for iter in range(n_iter):
            field_DOE = np.fft.ifft2(np.fft.ifftshift(field_image))                                                         # field DOE = TF-1 field image
            phase_DOE = np.angle(field_DOE)                                                                                 # get DOE phase. phase values between 0 and 2pi 
            phase_DOE = discretization(phase_DOE, n_levels)                                                                 # phase discretization
            field_DOE = np.exp(phase_DOE * 1j)                                                                              # force the amplitude of the DOE to 1 (no losses)
            field_image = np.fft.fftshift(np.fft.fft2(field_DOE))                                                           # image = TF du DOE
            phase_image = np.angle(field_image)                                                                             # save image phase
            amp_image[doe_size[0]//2-target_size[0]//2:doe_size[0]//2-target_size[0]//2+target_size[0], 
                      doe_size[1]//2-target_size[1]//2:doe_size[1]//2-target_size[1]//2+target_size[1]] = rfact*target_amp  # force the amplitude of the DOE to the target amplitude inside the ROI. Outside the ROI, the amplitude is free
            field_image = amp_image*np.exp(phase_image * 1j)                                                                # new image field computation

    if compute_efficiency:

        # Compute the image finally formed and the efficiency
    
        recovery = np.absolute(np.fft.fftshift(np.fft.fft2(field_DOE)))**2                                                  # Final image = |TF field DOE|^2
        efficiency = np.sum(recovery[doe_size[0]//2-target_size[0]//2:doe_size[0]//2-target_size[0]//2+target_size[0],      # efficiency = energy inside target zone / total energy
                  doe_size[1]//2-target_size[1]//2:doe_size[1]//2-target_size[1]//2+target_size[1]])/np.sum(recovery)
        
        return phase_DOE, recovery, efficiency

    return phase_DOE