# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 15:50:46 2024 under Python 3.11.7

Purpose : script that runs functions from hologram_design.py file across a big number of seeds
          in order to find the best result of the optimization process

@author: f24lerou
"""

# 8<------------------------------------------- Add path ---------------------------------------

import os
import sys

path = os.path.abspath(os.path.abspath('..'))
sys.path.append(path)

#%% 8<------------------------------ Directories and filenames --------------------------------

dirc = os.path.abspath(os.getcwd()) + r"\\"
dir_results_npy = dirc + r"results\npy\\"
dir_results_pgm = dirc + r"results\pgm\\"

#%% 8<-------------------------------------- Import modules -----------------------------------

import numpy as np
import cv2

from doe.hologram_design import GetOpticLengthMinMax, GetHoloSize

from doe.ifta import ifta, iftaSoftQuantization

from doe.tools import Replicate, discretization, PropagatePhaseScreen, PropagateComplexAmplitudeScreen

from doe.performance_criterias import ComputeEfficiency, ComputeUniformity

from doe.phaseScreens import lens

from doe.amplitudeScreens import Gaussian

from doe.gaussianBeams import getGaussianBeamRadius



from PIL import Image

import matplotlib.pyplot as plt

#%% 8<---------------------------------------- Parameters -------------------------------------

n_seeds = 500                  # number of seeds considered

n_iter = 500
rfact = 1.2

wavelength = 850e-9             # [m] wavelength - VSCEL: VC850S-SMD
divergence = 8                  # [°] gaussian beam divergence (full angle) - VSCEL: VC850S-SMD
 
d1 = 0.01                       # [m] distance laser object waist - holo
d2 = 0.03                       # [m] distance holo - image plane (image waist)


n_levels = 2                    # number of phase levels 

light_collection_efficiency_mini = 0.5  # minimal ratio between the energy emitted by the VCSEL and 
                                        # the incident energy on the hologram
        

fringe_length_mini = 2e-6       # [m] fabrication constaint : minimal width of the fringes at the edges of the
                                # Fresnel lens
                                
optic_length_factor = 0.2       # Choose optic_length between min and max

separation = 0.5e-3             # [m] step of the Dirac comb in image plane, i.e separation between 
                                # two samples dots
                                
optic_pp = 750e-9               # [m] pixel pitch on optic plane, imposed by the photoplotteur

n_points = 3

#%% 8<---------------------------------------- Consequences -------------------------------------

optic_length_mini, optic_length_maxi, f = GetOpticLengthMinMax(wavelength, divergence, d1, d2,
                                                      light_collection_efficiency_mini, fringe_length_mini)

optic_length = optic_length_mini + optic_length_factor*(optic_length_maxi - optic_length_mini)

holo_size, n_replications, optic_length, separation \
 = GetHoloSize(wavelength, d2, separation, optic_pp, optic_length)

target = np.zeros([holo_size]*2)

target[target.shape[0]//2-n_points//2:target.shape[0]//2+n_points//2+n_points%2,
              target.shape[1]//2-n_points//2:target.shape[1]//2+n_points//2+n_points%2] \
              = np.ones([n_points]*2)                               # padding the zone that will be multiplied by the dirac 
                                                                    # comb in image space with ones
                                                                    
#%% 8<-------------------------------------- hologram computation --------------------------------

              # Memory allocation #

phase_holo = np.full((holo_size, holo_size, 2*n_seeds), np.NAN)
phase_holo_sq = np.full((holo_size, holo_size, n_seeds), np.NAN)

seeds = 2*np.pi*np.random.rand(holo_size, holo_size, n_seeds) # starting point: random image phases



for k in range(n_seeds):
    
    seed = seeds[:,:,k]
    
    phase_holo[:,:,k] = ifta(target, target.shape, n_levels=n_levels, 
                                                            compute_efficiency=0, rfact=rfact, 
                                                            n_iter=n_iter, seed=seed) # ifta to compute hologram
                                                                                      # that will be replicated

    phase_holo[:,:,n_seeds+k] = iftaSoftQuantization(target, target.shape, n_levels=n_levels, 
                                                            compute_efficiency=0, rfact=rfact, 
                                                            n_iter=n_iter, seed=seed) # ifta to compute hologram
                                                                                      # that will be replicated
                                                                                      # with soft quantization
                                                                                      
#%% 8<----------------- Estimation of the performances ------------------------------------------

            # Memory allocation #

efficiency = np.full(2*n_seeds, np.NAN)
uniformity = np.full(2*n_seeds, np.NAN)
 
for k in range(2*n_seeds):
    
    efficiency[k] = ComputeEfficiency(phase_holo[:,:,k], target)
    uniformity[k] = ComputeUniformity(phase_holo[:,:,k], target)
    
#%%

phase_holo_replicated_efficiency = Replicate(phase_holo[:,:,np.where(efficiency==np.max(efficiency))[0][0]], n_replications)

phase_holo_replicated_uniformity = Replicate(phase_holo[:,:,np.where(uniformity==np.min(uniformity))[0][0]], n_replications)

most_efficient_uniformity = uniformity[np.where(efficiency==np.max(efficiency))[0][0]]

most_uniform_efficiency = efficiency[np.where(uniformity==np.min(uniformity))[0][0]]


#%% 8<----------------- Fresnel lens addition ------------------------------------------

phase_lens = lens(f, wavelength=wavelength, sizeSupport=phase_holo_replicated_efficiency.shape, 
                  samplingStep=optic_pp, n_levels=0)

phase_lens_discretized = discretization(phase_lens, n_levels=2)

phase_holo_replicated_efficiency_fresnel = discretization(phase_holo_replicated_efficiency + phase_lens, 
                                                          n_levels=2)

phase_holo_replicated_uniformity_fresnel = discretization(phase_holo_replicated_uniformity + phase_lens, 
                                                          n_levels=2)

#%% 8<-------------------- Save results ------------------------------------------------

# with phase values between 0 and 2pi under .npy file

np.save(dir_results_npy+"phase_lens_discretized.npy", phase_lens_discretized)

np.save(dir_results_npy+"phase_holo_replicated_efficiency", phase_holo_replicated_efficiency)
np.save(dir_results_npy+"phase_holo_replicated_uniformity", phase_holo_replicated_uniformity)

np.save(dir_results_npy+"phase_holo_replicated_efficiency_fresnel", phase_holo_replicated_efficiency_fresnel)
np.save(dir_results_npy+"phase_holo_replicated_uniformity_fresnel", phase_holo_replicated_efficiency)

#%% with phase values between 0 and 255 under .pgm file

phase_lens_discretized[phase_lens_discretized==np.pi] = 255

phase_holo_replicated_efficiency[phase_holo_replicated_efficiency==np.pi] = 255
phase_holo_replicated_uniformity[phase_holo_replicated_uniformity==np.pi] = 255

phase_holo_replicated_efficiency_fresnel[phase_holo_replicated_efficiency_fresnel==np.pi] = 255
phase_holo_replicated_uniformity_fresnel[phase_holo_replicated_uniformity_fresnel==np.pi] = 255

cv2.imwrite(dir_results_pgm+"phase_lens_discretized.pgm", np.asarray(phase_lens_discretized, dtype=np.uint8))

cv2.imwrite(dir_results_pgm+"phase_holo_replicated_efficiency.pgm", phase_holo_replicated_efficiency)
cv2.imwrite(dir_results_pgm+"phase_holo_replicated_uniformity.pgm", phase_holo_replicated_uniformity)

cv2.imwrite(dir_results_pgm+"phase_holo_replicated_efficiency_fresnel.pgm", phase_holo_replicated_efficiency_fresnel)
cv2.imwrite(dir_results_pgm+"phase_holo_replicated_uniformity_fresnel.pgm", phase_holo_replicated_uniformity_fresnel)

#%% Compute gaussian amplitude

w_z = getGaussianBeamRadius(wavelength=wavelength, divergence=divergence, propagation_distance=d1)
amplitude = Gaussian(phase_holo_replicated_efficiency.shape[0], pixel_pitch=optic_pp, sigma=w_z/2**0.5)

#%%

fig, axs = plt.subplots(nrows=2, ncols=4)

axs[0,0].axis("off")
axs[0,0].imshow(phase_holo_replicated_efficiency)
axs[0,0].set_title("hologram phase, best efficiency")

axs[1,0].axis("off")
axs[1,0].imshow(phase_holo_replicated_uniformity)
axs[1,0].set_title("hologram phase, best uniformity")


axs[0,1].axis("off")
axs[0,1].imshow(PropagatePhaseScreen(phase_holo_replicated_efficiency*np.pi/255.0))
axs[0,1].set_title("image formed, best efficiency")

axs[1,1].axis("off")
axs[1,1].imshow(PropagatePhaseScreen(phase_holo_replicated_uniformity))
axs[1,1].set_title("image formed, best uniformity")

axs[0,2].axis("off")
axs[0,2].imshow(np.log(PropagateComplexAmplitudeScreen(amplitude*np.exp(1j*phase_holo_replicated_efficiency*np.pi/255.0))+1))
axs[0,2].set_title("image formed, best efficiency\nlog scale and gaussian amplitude")

axs[1,2].axis("off")
axs[1,2].imshow(np.log(PropagateComplexAmplitudeScreen(amplitude*np.exp(1j*phase_holo_replicated_uniformity*np.pi/255.0))+1))
axs[1,2].set_title("image formed, best uniformity\nlog scale and gaussian amplitude")

axs[0,3].axis("off")
axs[0,3].imshow(phase_holo_replicated_efficiency_fresnel)
axs[0,3].set_title("hologram phase\nbest efficiency + Fresnel lens")

axs[1,3].axis("off")
axs[1,3].imshow(phase_holo_replicated_uniformity_fresnel)
axs[1,3].set_title("hologram phase\nbest uniformity + Fresnel lens")



