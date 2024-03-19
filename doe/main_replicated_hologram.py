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

from doe.tools import discretization, getCartesianCoordinates
from doe.gaussianBeams import getGaussianBeamRadius, getCollectorLengthMini, getFocalLength, \
                              gaussianEfficiency, getImageWaist, divergenceToWaist
from doe.ifta import ifta, iftaSoftQuantization

#%% 8<------------------------------------ Parameters ------------------------------------------

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

optic_length_factor = 0.3
optic_length = optic_length_mini + optic_length_factor*(optic_length_maxi - optic_length_mini)

#%% 8<--------------------------------- Replication workflow --------------------------------------

n_points = 3

separation = 100e-6 #2*w_0_prime # [m] step of the Dirac comb in image plane, i.e separation between two samples dots

replication_step = wavelength * d2 / separation # [m] step of the Dirac comb in optic space, i.e separation
                                                # between two replicated holograms, i.e hologram side length

holo_size = np.array([replication_step//optic_pp - (replication_step//optic_pp)%2]*2, dtype=int)                                                

n_replications = int(optic_length//replication_step)

target_suport = np.zeros(holo_size) # support of the target image

target_pp = wavelength * d2 / replication_step # optic_length or replication_step at the denominator ?

target_length = n_points*separation # [m]

target_size = np.array([target_length//target_pp + np.ceil(target_length%target_pp)]*2, dtype=int)

target_suport[target_suport.shape[0]//2-target_size[0]//2:target_suport.shape[0]//2+target_size[0]//2+1,
              target_suport.shape[1]//2-target_size[1]//2:target_suport.shape[1]//2+target_size[1]//2+1] = np.ones(target_size)

phase_holo, recovery, efficiency = iftaSoftQuantization(target_suport, target_suport.shape, n_levels=n_levels, compute_efficiency=1, 
                                       rfact=1.2, n_iter=100)


#%%
phase_holo_replicated = np.full(n_replications*holo_size, np.nan)

for i in range(n_replications):
    for j in range(n_replications):
            phase_holo_replicated[i*holo_size[0]:(i+1)*holo_size[0],
                                  j*holo_size[1]:(j+1)*holo_size[1]] = phase_holo

#%%

image_pp = wavelength * d2 / optic_length # [m] pixel pitch on image plane

image_holo = np.abs(np.fft.fftshift(np.fft.fft2(np.exp(1j*phase_holo))))**2
image_holo_replicated = np.abs(np.fft.fftshift(np.fft.fft2(np.exp(1j*phase_holo_replicated))))**2

holo_window = 20
holo_replicated_window = int((n_points+10)*separation//image_pp)

image_holo_max = np.where(image_holo==np.max(image_holo))
image_holo_cropped = image_holo[image_holo_max[0][0]-holo_window//2:image_holo_max[0][0]+holo_window//2, 
                                image_holo_max[1][0]-holo_window//2:image_holo_max[1][0]+holo_window//2]


image_holo_replicated_max = np.where(image_holo_replicated==np.max(image_holo_replicated))
image_holo_replicated_cropped = image_holo_replicated[image_holo_replicated_max[0][0]-
                                                      holo_replicated_window//2 : image_holo_replicated_max[0][0]+
                                                      holo_replicated_window//2, 
                                image_holo_replicated_max[1][0]-holo_replicated_window//2:image_holo_replicated_max[1][0]+
                                holo_replicated_window//2]

# %%


[X,Y] = getCartesianCoordinates(image_holo_cropped.shape[0])
x_axis = image_pp * X[0,:]
y_axis = image_pp * Y[:,0]

fig, axs = plt.subplots(nrows=2, ncols=2)
axs[0,0].imshow(phase_holo)
axs[0,1].imshow(phase_holo_replicated)
axs[1,0].imshow(image_holo_cropped)

axs[1,1].imshow(image_holo_replicated_cropped, extent=                                   # [µm]
                        1e6*np.array([x_axis[0], x_axis[-1], 
                                      y_axis[-1], y_axis[0]]))
axs[1,1].set_xlabel("[µm]")
axs[1,1].set_ylabel("[µm]")


# target = gridSquares(nPointsX = 3, spacing=1, width=1)

# target_length = target.shape[0] * image_pp

# holo_length = 1/image_pp # [m] side length of one hologram

# holo_size = np.array([holo_length//optic_pp + (holo_length//optic_pp)%2]*2)















