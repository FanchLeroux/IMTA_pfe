# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 09:14:53 2024 under Python 3.11.7

@author: f24lerou
"""
# 8<------------------------------------------ Add path ---------------------------------------

import os
import sys

path = os.path.abspath(os.path.abspath('..')) # path = 'D:\\francoisLeroux\\codes'
sys.path.append(path)

#%% 8<------------------------------ Directories and filenames --------------------------------

dirc = os.path.abspath(os.getcwd()) + r"\\"
paternFilename = dirc + r"patern\outputs\\"+"cross_5_32x32.npy"
dir_results = dirc + r"results\\"

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

#%% 8<------------------------------------ Parameters -------------------------------------------

make_plot = 1  # make plot
sq = 0         # use soft quantization

                         ################# Requierments ###################
        
# geometry        
d1 = 0.01                                   # [m] distance laser object waist - holo
d2 = 0.03                                   # [m] distance holo - image plane (image waist)

# number of phase levels
n_levels = 2

# limits
light_collection_efficiency_mini = 0.5      # minimal ratio between the energy emitted by the VCSEL and 
                                            # the incident energy on the hologram
                         
                         ############ Constraints from hardware ############
        
wavelength = 850e-9             # [m] wavelength - VSCEL: VC850S-SMD
divergence = 8                  # [°] gaussian beam divergence (full angle) - VSCEL: VC850S-SMD
fringe_length_mini = 2e-6       # [m] fabrication constaint : minimal width of the fringes at the edges of the
                                # fresnel lens (half a period)
optic_pp = 750e-9               # [m] pixel pitch on optic plane, imposed by the photoplotteur

                         ################# Consequences ####################

# Fresnel lens focal length
[f, diff] = getFocalLength(d1, d2, wavelength, divergence) # focal length for source - image plane conjugation

# optic side length mini to match light collection requirement
w_z = getGaussianBeamRadius(wavelength=wavelength, divergence=divergence, propagation_distance=d1)
optic_length_mini = getCollectorLengthMini(w_z=w_z, efficiency=light_collection_efficiency_mini)

# optic side length maxi to match thin fringes requirement
optic_length_maxi = getOpticSideLengthMaxi(wavelength, f, fringe_length_mini)

# light collection maxi
light_collection_efficiency_maxi = gaussianEfficiency(wavelength, d1, optic_length_maxi/2, divergence=divergence)

# get object and image waist
w_0 = divergenceToWaist(wavelength, divergence)
w_0_prime = getImageWaist(wavelength, f, w_0, d1)

#%% choose optic_length 

optic_length_factor = 0.2
optic_length = optic_length_mini + optic_length_factor*(optic_length_maxi - optic_length_mini)

#%% 8<--------------------------------- Replication workflow --------------------------------------

n_points = 5

separation = 0.5e-3                         # [m] step of the Dirac comb in image plane, i.e separation between 
                                            # two samples dots

holo_length = wavelength * d2 / separation  # [m] step of the Dirac comb in optic space, i.e separation
                                            # between two replicated holograms, i.e hologram side length

holo_size = np.array([holo_length//optic_pp - (holo_length//optic_pp)%2]*2, dtype=int)                                                

n_replications = int(optic_length//holo_length)
optic_length = n_replications * holo_length # [m] final optic side length

target_suport = np.zeros(holo_size)         # support of the target image. same size as holo

target_pp = wavelength * d2 / holo_length   # [m] pixel pitch in image plane without replication

target_length = n_points*separation         # [m] length of the zone in target that should be filled with ones (see next lines)

target_size = np.array([target_length//target_pp + np.ceil(target_length%target_pp)]*2, dtype=int) # [px]

target_suport[target_suport.shape[0]//2-target_size[0]//2:target_suport.shape[0]//2+target_size[0]//2+target_size[0]%2,
              target_suport.shape[1]//2-target_size[1]//2:target_suport.shape[1]//2+target_size[1]//2+target_size[1]%2] \
              = np.ones(target_size)                               # padding the zone that will be multiplied by the dirac 
                                                                   # comb in image space with ones

#%% 8<----------------------------------------- IFTA ---------------------------------------------

if not(sq):
    phase_holo, recovery, efficiency = ifta(target_suport, target_suport.shape, n_levels=n_levels, 
                                                            compute_efficiency=1, rfact=1.2, 
                                                            n_iter=20) # ifta to compute hologram
                                                                       # that will be replicated
                                                                       
    phase_holo_replicated = np.full(n_replications*holo_size, np.nan)

    for i in range(n_replications):
        for j in range(n_replications):
                phase_holo_replicated[i*holo_size[0]:(i+1)*holo_size[0],
                                      j*holo_size[1]:(j+1)*holo_size[1]] = phase_holo
                                                                   
else:
    phase_holo_sq, recovery_sq, efficiency_sq = iftaSoftQuantization(target_suport, target_suport.shape, n_levels=n_levels, 
                                                            compute_efficiency=1, rfact=1.2, 
                                                            n_iter=20) # ifta with phase soft quantization to compute hologram
                                                                       # that will be replicated

    phase_holo_sq_replicated = np.full(n_replications*holo_size, np.nan)

    for i in range(n_replications):
        for j in range(n_replications):
                phase_holo_sq_replicated[i*holo_size[0]:(i+1)*holo_size[0],
                                      j*holo_size[1]:(j+1)*holo_size[1]] = phase_holo_sq

#%% 8<------------------------------------------------ Plots ---------------------------------------------------

if make_plot:
    if not(sq):

        ############################################ Physical Units ####################################################

        image_pp = wavelength * d2 / (phase_holo_replicated.shape[0]*optic_pp)         # [m] pixel pitch on image plane       

        image_holo_sq_window = int(2*target_size[0])
        image_holo_sq_replicated_window = 2*int(separation * n_points // image_pp)

        ######################################## Fresnel lens addition #################################################

        phase_lens = lens(f, wavelength=wavelength, sizeSupport=phase_holo_replicated.shape, samplingStep=optic_pp, n_levels=0)
        phase_holo_replicated_fresnel = discretization(phase_holo_replicated + phase_lens, n_levels=2)

        
        ###################################### Light Propagation ###############################################

        image_holo = np.abs(np.fft.fftshift(np.fft.fft2(np.exp(1j*phase_holo))))**2  # image given by a single hologram
                                                                                     # under plane wave illumination
        image_holo_cropped = image_holo[image_holo.shape[0]//2-image_holo_sq_window:
                                        image_holo.shape[0]//2+image_holo_sq_window, 
                                        image_holo.shape[0]//2-image_holo_sq_window:
                                        image_holo.shape[0]//2+image_holo_sq_window]

        amplitude = Gaussian(size_support=phase_holo_replicated.shape[0], pixel_pitch=optic_pp, sigma=w_z/2**0.5)
    
        field = amplitude*np.exp(1j*phase_holo_replicated)
        image_holo_replicated = np.abs(np.fft.fftshift(np.fft.fft2(field)))**2 # image formed when replicating hologram
            
        image_holo_replicated_cropped = image_holo_replicated[image_holo_replicated.shape[0]//2-
                                        image_holo_sq_replicated_window//2 : image_holo_replicated.shape[0]//2+
                                        image_holo_sq_replicated_window//2, 
                                        image_holo_replicated.shape[0]//2-image_holo_sq_replicated_window//2:
                                        image_holo_replicated.shape[0]//2+image_holo_sq_replicated_window//2]
        
        [X,Y] = getCartesianCoordinates(phase_holo_replicated.shape[0])
        x_axis = optic_pp * X[0,:]
        y_axis = optic_pp * Y[:,0]
            
        [X,Y] = getCartesianCoordinates(image_holo_replicated_cropped.shape[0])
        x_axis_cropped = image_pp * X[0,:]
        y_axis_cropped = image_pp * Y[:,0]
        
        fig, axs = plt.subplots(nrows=2, ncols=3)
        
        axs[0,0].imshow(phase_holo)
        axs[0,0].set_title("phase, one period")
        
        axs[0,1].imshow(phase_holo_replicated, extent=               # [µm]
                                1e6*np.array([x_axis[0], x_axis[-1],
                                              y_axis[-1], y_axis[0]]))
        axs[0,1].set_title("phase, replicated hologram")
        axs[0,1].set_xlabel("[µm]")        
        axs[0,1].set_ylabel("[µm]")
        
        axs[0,2].imshow(phase_holo_replicated_fresnel, extent=               # [µm]
                                1e6*np.array([x_axis[0], x_axis[-1],
                                              y_axis[-1], y_axis[0]]))
        axs[0,2].set_title("phase, replicated hologram and Fresnel lens")
        axs[0,2].set_xlabel("[µm]")        
        axs[0,2].set_ylabel("[µm]")
        
        axs[1,0].imshow(image_holo_cropped)
        axs[1,0].set_title("image formed by one period")
        
        axs[1,1].imshow(image_holo_replicated_cropped+1, extent=               # [µm]
                                1e6*np.array([x_axis_cropped[0], x_axis_cropped[-1],
                                              y_axis_cropped[-1], y_axis_cropped[0]]))
        axs[1,1].set_xlabel("[µm]")
        axs[1,1].set_ylabel("[µm]")
        axs[1,1].set_title("log scale, gaussian amplitude propagation\nthrough replicated hologram")
        
        axs[1,2].plot(1e6*x_axis_cropped, image_holo_replicated_cropped[:,image_holo_replicated_cropped.shape[1]//2])
        axs[1,2].set_xlabel("[µm]")
        axs[1,2].set_title("crossY")
        
    else:
        
        ############################################ Physical Units ####################################################

        image_pp = wavelength * d2 / (phase_holo_replicated.shape[0]*optic_pp)         # [m] pixel pitch on image plane
        
        image_holo_sq_window = int(2*target_size[0])
        image_holo_sq_replicated_window = 2*int(separation * n_points // image_pp)
        
        ######################################## Fresnel lens addition #################################################

        phase_lens = lens(f, wavelength=wavelength, sizeSupport=phase_holo_sq_replicated.shape, samplingStep=optic_pp, n_levels=0)
        phase_holo_sq_replicated_fresnel = discretization(phase_holo_sq_replicated + phase_lens, n_levels=2)
        
        ############################################# Else ############################################################
        
        image_holo_sq = np.abs(np.fft.fftshift(np.fft.fft2(np.exp(1j*phase_holo_sq))))**2 # image given by a single soft-quantized
                                                                                          # hologram under plane wave illumination
        
        amplitude = Gaussian(size_support=phase_holo_sq_replicated.shape[0], pixel_pitch=optic_pp, sigma=w_z/2**0.5)
        
        field_sq = amplitude*np.exp(1j*phase_holo_sq_replicated)
        image_holo_sq_replicated = np.abs(np.fft.fftshift(np.fft.fft2(field_sq)))**2 # image formed when replicating hologram 
                                                                                     # obtained with soft quantization
    
        image_holo_sq_cropped = image_holo_sq[image_holo_sq.shape[0]//2-image_holo_sq_window:
                                        image_holo_sq.shape[0]//2+image_holo_sq_window, 
                                        image_holo_sq.shape[0]//2-image_holo_sq_window:
                                        image_holo_sq.shape[0]//2+image_holo_sq_window]
    
        image_holo_sq_replicated_cropped = image_holo_sq_replicated[image_holo_sq_replicated.shape[0]//2-
                                        image_holo_sq_replicated_window//2 : image_holo_sq_replicated.shape[0]//2+
                                        image_holo_sq_replicated_window//2, 
                                        image_holo_sq_replicated.shape[0]//2-image_holo_sq_replicated_window//2:
                                        image_holo_sq_replicated.shape[0]//2+image_holo_sq_replicated_window//2]
                                                                 
        [X,Y] = getCartesianCoordinates(image_holo_sq_replicated_cropped.shape[0])
        x_axis_cropped = image_pp * X[0,:]
        y_axis_cropped = image_pp * Y[:,0]
        
        fig, axs = plt.subplots(nrows=2, ncols=3)
        
        axs[0,0].imshow(phase_holo_sq)
        
        axs[0,1].imshow(phase_holo_sq_replicated)
        
        axs[0,2].axis('off')
        
        axs[1,0].imshow(image_holo_sq_cropped)
        
        axs[1,1].imshow(image_holo_sq_replicated_cropped, extent=               # [µm]
                                1e6*np.array([x_axis_cropped[0], x_axis_cropped[-1],
                                              y_axis_cropped[-1], y_axis_cropped[0]]))
        axs[1,1].set_xlabel("[µm]")
        axs[1,1].set_ylabel("[µm]")
        
        axs[0,2].plot(1e6*x_axis_cropped, image_holo_sq_replicated_cropped[:,image_holo_sq_replicated_cropped.shape[1]//2])
        axs[0,2].set_xlabel("[µm]")
        axs[0,2].set_title("with soft quantization")
        
        axs[1,2].plot(1e6*x_axis_cropped, image_holo_replicated_cropped[:,image_holo_sq_replicated_cropped.shape[1]//2])
        axs[1,2].set_xlabel("[µm]")
        axs[1,2].set_title("no soft quantization")

plt.tight_layout()





