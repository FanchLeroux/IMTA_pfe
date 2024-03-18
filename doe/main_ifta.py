# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 10:07:22 2024 under Python 3.11.7

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
from doe.gaussianBeams import getGaussianBeamRadius, getCollectorLengthMini, getFocalLength, gaussianEfficiency
from doe.ifta import ifta, iftaSoftQuantization

#%% 8<------------------------------------ Parameters ------------------------------------------

                         ################# Requierments ###################
        
# geometry        
d1 = 0.01                                   # [m] distance laser object waist - holo
d2 = 0.03                                   # [m] distance holo - image plane (image waist)

# number of phase levels
n_levels = 2

# replication
n_replication = 2                           # number of time the hologram should be replicated in X and Y direction

# limits
light_collection_efficiency_mini = 0.5      # minimal ratio between the energy emitted by the VCSEL and the incident energy on the holo
holo_efficiency_mini = 0.7                  # minimal holo efficiency (ratio energy in ROI / total energy in image plane) 

                         ########## Constraints from hardware #############
        
wavelength = 850e-9             # [m] wavelength - VSCEL: VC850S-SMD
divergence = 10                 # [°] gaussian beam divergence (full angle) - VSCEL: VC850S-SMD
fringe_length_mini = 2e-6       # [m] fabrication constaint : minimal width of the fringes at the edges of the
                                # fresnel lens (half a period)
optic_pp = 750e-9               # [m] pixel pitch on optic plane, imposed by the fabrication process

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

# target length maxi to avoid isolated pixels on doe
target_length_maxi = wavelength*d2/(2*optic_pp) # [m] delta_f max = 1/direct_space_pp. Division by
                                                # two to avoid isolated pixels

                         ################### Arbitrage #######################

#optic_length = 1.1*optic_length_mini
#optic_length = 0.9*optic_length_maxi
optic_length = optic_length_mini + 0.5*(optic_length_maxi - optic_length_mini)
holo_length = optic_length/n_replication                 # [m] hologram side length. Will be replicated
                                                         # n_replication * n_replication times 
holo_size = int(holo_length//optic_pp)                   # [px] hologram size
holo_size = np.array([holo_size + holo_size%2]*2)                  # [px] hologram size (even)
holo_length = holo_size[0] * optic_pp                    # [m] hologram side length after sampling
optic_size = n_replication * holo_size                   # [px] optic size
optic_length = n_replication * holo_length               # [m] optic side length after sampling


light_collection_efficiency = gaussianEfficiency(wavelength, d1, optic_length/2, divergence=divergence)

target_length = 0.9 * target_length_maxi                 # [m] target side length

image_pp = wavelength * d2 * 1/holo_length               # [m] pixel pitch in image plane

# target image
target_size = int(target_length//image_pp)               # [px] image size

#%% 8<--------------------------------------- main -------------------------------------------

# target definition
#target = cross(cross_size=target_size, width=width, support_size = [target_size+10,target_size+10]) # cross
target = gridSquares(nPointsX = 3, spacing=target_size//5, width=target_size//5)

# ifta - no soft quantization
phase_holo, recovery, efficiency = ifta(target, holo_size, n_levels=n_levels, compute_efficiency=1, 
                                       rfact=1.2, n_iter=100)

# ifta - phase soft quantization
phase_holo_soft, recovery_soft, efficiency_soft = iftaSoftQuantization(target, holo_size, n_levels=n_levels, 
                                                                      compute_efficiency=1, rfact=1.2, n_iter=100)

#%% Fresnel lens computation
phase_lens = lens(f, wavelength=wavelength, sizeSupport=optic_size, samplingStep=optic_pp, n_levels=0)
phase_holo_lens = phase_holo + phase_lens[:holo_size[0], :holo_size[1]] # adding top left corner of the fresnel lens
phase_holo_lens_discretized = discretization(phase_holo_lens, n_levels)

#%% 8<----------------------------------- results -----------------------------------------

np.save(dir_results+"holo", phase_holo)
np.save(dir_results+"holoLens", phase_holo_lens)

#%% 8<------------------------------- param text file --------------------------------------

params = ["light collection requierment",
          "wavelength [nm]", 
          "d1 [cm]", 
          "d2 [cm]",
          "focal length [mm]",         
          
          "relative difference between thin lens formula \n\tand modified thin lens formula",
          "target image diameter maxi to avoid isolated pixels [mm]",
          "target image diameter [mm]",
          "number of replication in X and Y",
          "optic side length mini (light collection requierment) [µm]",
          
          "optic side length maxi (thin fringes requierments) [µm]",
          "light collection efficiency maxi",
          "optic side length [µm]",
          "optic pixel pitch [nm]",
          "hologram side length [µm]",
          
          "hologram diameter [px]",
          "light collection efficiency",
          "hologram efficiency - no soft quantization",
          "hologram efficiency - with phase soft quantization"]

elts = [str(light_collection_efficiency_mini) + "\n",
        str(wavelength*1e9) + "\n", 
        str(100*d1), 
        str(100*d2),
        str(np.round(f*1e3, decimals=1)),
        
        str(np.round(diff/f, decimals=6)) + "\n",
        str(np.round(target_length_maxi*1e3, decimals=2)),
        str(np.round(target_size*image_pp*1e3, decimals=2)) + "\n",
        str(n_replication) + "\n",
        str(np.round(optic_length_mini*1e6, decimals=1)),
        
        str(np.round(optic_length_maxi*1e6, decimals=1)) + " => light collection efficiency maxi",
        str(np.round(light_collection_efficiency_maxi, decimals=2)) + "\n",
        str(np.round(optic_length*1e6, decimals=1)),        
        str(np.round(optic_pp*1e9, decimals=1)) + "\n",
        str(np.round(holo_length*1e6, decimals=2)),
    
        str(holo_size[0]) + "\n",
        str(np.round(light_collection_efficiency, decimals=2)) + "\n",
        str(np.round(efficiency, decimals=4)),
        str(np.round(efficiency_soft, decimals=4))]

with open(dir_results+'params.txt', 'w') as file:
    file.write("\n\n")
    for k in range(len(params)):        
            file.write("\t" + params[k] + " : " + elts[k] + "\n")

#%% 8<------------------------ ----- plots in physical units ------------------------------------

[X,Y] = getCartesianCoordinates(nrows=holo_size[0])
x_axis_image_plane = image_pp * X[0,:]                    # [m]
y_axis_image_plane = image_pp * Y[:,0]                    # [m]
x_axis_holo_plane = optic_pp * X[0,:]                     # [m]
y_axis_holo_plane = optic_pp * Y[:,0]                     # [m]

[X,Y] = getCartesianCoordinates(nrows=target.shape[0])
x_axis_target = image_pp * X[0,:]                         # [m]
y_axis_target = image_pp * Y[:,0]                         # [m]

fig2, axs2 = plt.subplots(nrows=2, ncols=3)

fig200=axs2[0,0].imshow(target, aspect="equal", extent=                       # [cm]
                        1e2*np.array([x_axis_target[0], x_axis_target[-1], 
                                      y_axis_target[-1], y_axis_target[0]]))
axs2[0,0].set_title("target")
axs2[0,0].set_xlabel("[cm]")
axs2[0,0].set_ylabel("[cm]")

fig201=axs2[0,1].imshow(phase_holo, extent=                                   # [µm]
                        1e6*np.array([x_axis_holo_plane[0], x_axis_holo_plane[-1], 
                                      y_axis_holo_plane[-1], y_axis_holo_plane[0]]))
axs2[0,1].set_title("phase_holo ("+str(n_levels)+" levels)")
axs2[0,1].set_xlabel("[µm]")
axs2[0,1].set_ylabel("[µm]")
divider = make_axes_locatable(axs2[0,1])
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(fig201, cax=cax)

fig202=axs2[0,2].imshow(recovery, extent=                                     # [mm]
                        1e3*np.array([x_axis_image_plane[0], x_axis_image_plane[-1], 
                                      y_axis_image_plane[-1], y_axis_image_plane[0]]))
axs2[0,2].set_title("Image plane - Irradiance\n efficiency = "+str(round(efficiency*100))+"%")
axs2[0,2].set_xlabel("[mm]")
axs2[0,2].set_ylabel("[mm]")
divider = make_axes_locatable(axs2[0,2])
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(fig202, cax=cax)

fig210=axs2[1,0].imshow(phase_lens, extent=                                   # [µm]
                        1e6*np.array([x_axis_holo_plane[0], x_axis_holo_plane[-1], 
                                      y_axis_holo_plane[-1], y_axis_holo_plane[0]]))
axs2[1,0].set_title("phase_lens")
axs2[1,0].set_xlabel("[µm]")
axs2[1,0].set_ylabel("[µm]")
divider = make_axes_locatable(axs2[1,0])
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(fig210, cax=cax)

fig211=axs2[1,1].imshow(phase_holo_lens, extent=                              # [µm]
                        1e6*np.array([x_axis_holo_plane[0], x_axis_holo_plane[-1], 
                                      y_axis_holo_plane[-1], y_axis_holo_plane[0]]))
axs2[1,1].set_title("phase_holo + top left corner of phase lens")
axs2[1,1].set_xlabel("[µm]")
axs2[1,1].set_ylabel("[µm]")
divider = make_axes_locatable(axs2[1,1])
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(fig211, cax=cax)

fig212=axs2[1,2].imshow(phase_holo_lens_discretized, extent=                  # [µm]
                        1e6*np.array([x_axis_holo_plane[0], x_axis_holo_plane[-1], 
                                      y_axis_holo_plane[-1], y_axis_holo_plane[0]]))
axs2[1,2].set_title("phase_holo_lens_discretized")
axs2[1,2].set_xlabel("[µm]")
axs2[1,2].set_ylabel("[µm]")
divider = make_axes_locatable(axs2[1,2])
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(fig212, cax=cax)

plt.tight_layout()