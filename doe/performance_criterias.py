# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 15:57:04 2024

@author: f24lerou
"""

# 8<------------------------------------------- Add path ---------------------------------------

import os
import sys

path = os.path.abspath(os.path.abspath('..'))
sys.path.append(path)

#%% 8<-------------------------------------- Import modules -----------------------------------

import numpy as np

#%% 8<--------------------------------------- Functions definitions ------------------------------

def ComputeEfficiency(phase_holo, target):
    
    recovery = np.absolute(np.fft.fftshift(np.fft.fft2(np.exp(1j*phase_holo))))**2 # Final image = |TF field DOE|^2
    efficiency = np.sum(recovery[target!=0])/np.sum(recovery)
    
    return efficiency

def ComputeUniformity(phase_holo, target):
    
    recovery = np.absolute(np.fft.fftshift(np.fft.fft2(np.exp(1j*phase_holo))))**2 # Final image = |TF field DOE|^2
    recovery = recovery[target!=0]
    uniformity = (np.max(recovery)-np.min(recovery))/(np.max(recovery)+np.min(recovery))
    
    return uniformity


# ====================== Klauss metric=====================================
# def ComputeUniformity(phase_holo, target):
#     
#     recovery = np.absolute(np.fft.fftshift(np.fft.fft2(np.exp(1j*phase_holo))))**2 # Final image = |TF field DOE|^2
#     recovery = recovery[target!=0]
#     mean_recovery = np.mean(recovery)
#     uniformity = (1-np.sum((recovery-mean_recovery)**2))/(recovery.size * mean_recovery)
#     
#     return uniformity
# =============================================================================
