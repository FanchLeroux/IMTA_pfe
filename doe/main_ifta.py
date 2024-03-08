# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 10:07:22 2024 under Python 3.11.7

@author: f24lerou
"""

# 8<---------------------------- Add path --------------------------------

import os
import sys

path = os.path.abspath(os.path.abspath('..'))
sys.path.append(path)

# 8<--------------------------- Import modules ---------------------------

from mpl_toolkits.axes_grid1 import make_axes_locatable

import numpy as np
import matplotlib.pyplot as plt

from doe.paterns import cross

from doe.phaseScreens import lens

from doe.tools import computeFocal, discretization, getCartesianCoordinates
from doe.gaussianBeams import getGaussianBeamRadius, getMinimalCollectorLength
from doe.ifta import ifta, iftaSoftQuantization



# 8<---------------- Directories and filenames ---------------------------

dirc = os.path.abspath(os.getcwd()) + "/"
paternFilename = dirc + r"patern\outputs\\"+"cross_5_32x32.npy"
dir_results = dirc + r"results\\"

# 8<--------------------- Parameters -------------------------------------

        ################# Requierments ###################
        
# geometry        
d1 = 0.01                                   # [m] distance laser - DOE
d2 = 0.03                                   # [m] distance DOE - image plane
target_length = 0.003                       # [m] target side length
target_width = target_length/5              # [m] target width

# limits
light_collection_efficiency_mini = 0.5      # minimal ratio between the energy emitted by the VCSEL and the incident energy on the DOE
doe_efficiency_mini = 0.7                   # minimal doe efficiency (ratio energy in ROI / total energy in image plane) 

        ########## Constraints from hardware #############
        
wavelength = 850e-9             # [m] wavelength - VSCEL: VC850S-SMD
divergence = 8                  # [°] gaussian beam divergence (full angle) - VSCEL: VC850S-SMD
fringe_length_mini = 2e-6       # [m] fabrication constaint minimal width of the fringes at the edges of the fresnel lens
optic_pp = 750e-9               # [m] pixel pitch on optic plane, imposed by the fabrication process


        ################# Consequences ####################

# laser waist
w_z = getGaussianBeamRadius(wavelength=wavelength, divergence=divergence, propagation_distance=d1)
optic_length_mini = getMinimalCollectorLength(w_z=w_z, efficiency=light_collection_efficiency_mini)

        ################### Arbitrage #######################

optic_length = 1.1*optic_length_mini
n_replication = 2
doe_length = optic_length/n_replication                 # [m] doe side length, n_replication x n_replication replication 
doe_size = [int(doe_length//optic_pp)]*2                # [px] doe size
doe_length = doe_size[0] * optic_pp                     # [m] doe side length after sampling
optic_length = n_replication * doe_length               # [m] optic side length after sampling


image_pp = wavelength * d2 * 1/doe_length               # [m] pixel pitch in image plane

# target image
target_size = int(target_length//image_pp)              # [px] image size
width = int(target_width//image_pp)                     # [px] image size

n_levels = 2

# 8<--------------------- main -------------------------------------------

target = cross(cross_size=target_size, width=width, support_size = [target_size+10,target_size+10])

phase_doe, recovery, efficiency = ifta(target, doe_size, n_levels=n_levels, compute_efficiency=1, rfact=1.2, n_iter=100)
phase_doe_soft, recovery_soft, efficiency_soft = iftaSoftQuantization(target, doe_size, n_levels=n_levels, compute_efficiency=1, rfact=1.2, n_iter=100)

f = computeFocal(d1, d2) # focal length for source - image plane conjugation
phase_lens = lens(f, wavelength=wavelength, sizeSupport=doe_size, samplingStep=doe_length/doe_size[0], n_levels=0)

phase_doe_lens = phase_doe + phase_lens
phase_doe_lens_discretized = discretization(phase_doe+phase_lens, n_levels)

# 8<-------------------- results -----------------------------------------

np.save(dir_results+"crossDoe", phase_doe)
np.save(dir_results+"crossDoeLens", phase_doe_lens)


print("\noptic_length_mini = "+str(round(optic_length_mini*1e6, ndigits=1))+" µm")
print("optic_length = "+str(round(optic_length*1e6, ndigits=1))+" µm")
print("doe_length = "+str(round(doe_length*1e6, ndigits=1))+" µm")
print("doe_size = "+str(doe_size[0])+" px\n")

print("optic_pp = "+str(round(optic_pp*1e9))+" nm")
print("image_pp = "+str(round(image_pp*1e6))+" µm\n")

print("efficiency = "+str(efficiency))
print("efficiency_soft = "+str(efficiency_soft))

#%% 8<-------------------- plots in physical units ------------------------

pp_doe_plane = doe_length/doe_size[0]               # [m]
pp_image_plane = wavelength * d2 * 1/doe_length     # [m]

[X,Y] = getCartesianCoordinates(nrows=doe_size[0])
x_axis_image_plane = pp_image_plane * X[0,:]        # [m]
y_axis_image_plane = pp_image_plane * Y[:,0]        # [m]
x_axis_doe_plane = pp_doe_plane * X[0,:]            # [m]
y_axis_doe_plane = pp_doe_plane * Y[:,0]            # [m]

[X,Y] = getCartesianCoordinates(nrows=target.shape[0])
x_axis_target = pp_image_plane * X[0,:]             # [m]
y_axis_target = pp_image_plane * Y[:,0]             # [m]

fig2, axs2 = plt.subplots(nrows=2, ncols=3)

fig200=axs2[0,0].imshow(target, aspect="equal", extent=                       # [cm]
                        1e2*np.array([x_axis_target[0], x_axis_target[-1], y_axis_target[-1], y_axis_target[0]]))
axs2[0,0].set_title("target")
axs2[0,0].set_xlabel("[cm]")
axs2[0,0].set_ylabel("[cm]")

fig201=axs2[0,1].imshow(phase_doe, extent=                                    # [µm]
                        1e6*np.array([x_axis_doe_plane[0], x_axis_doe_plane[-1], y_axis_doe_plane[-1], y_axis_doe_plane[0]]))
axs2[0,1].set_title("phase_doe ("+str(n_levels)+" levels)")
axs2[0,1].set_xlabel("[µm]")
axs2[0,1].set_ylabel("[µm]")
divider = make_axes_locatable(axs2[0,1])
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(fig201, cax=cax)

fig202=axs2[0,2].imshow(recovery, extent=                                     # [mm]
                        1e3*np.array([x_axis_image_plane[0], x_axis_image_plane[-1], y_axis_image_plane[-1], y_axis_image_plane[0]]))
axs2[0,2].set_title("Image plane - Irradiance\n efficiency = "+str(round(efficiency*100))+"%")
axs2[0,2].set_xlabel("[mm]")
axs2[0,2].set_ylabel("[mm]")
divider = make_axes_locatable(axs2[0,2])
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(fig202, cax=cax)

fig210=axs2[1,0].imshow(phase_lens, extent=                                   # [µm]
                        1e6*np.array([x_axis_doe_plane[0], x_axis_doe_plane[-1], y_axis_doe_plane[-1], y_axis_doe_plane[0]]))
axs2[1,0].set_title("phase_lens")
axs2[1,0].set_xlabel("[µm]")
axs2[1,0].set_ylabel("[µm]")
divider = make_axes_locatable(axs2[1,0])
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(fig210, cax=cax)

fig211=axs2[1,1].imshow(phase_doe_lens, extent=                               # [µm]
                        1e6*np.array([x_axis_doe_plane[0], x_axis_doe_plane[-1], y_axis_doe_plane[-1], y_axis_doe_plane[0]]))
axs2[1,1].set_title("phase_doe_lens ("+str(n_levels)+" levels)")
axs2[1,1].set_xlabel("[µm]")
axs2[1,1].set_ylabel("[µm]")
divider = make_axes_locatable(axs2[1,1])
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(fig211, cax=cax)

fig212=axs2[1,2].imshow(phase_doe_lens_discretized, extent=                   # [µm]
                        1e6*np.array([x_axis_doe_plane[0], x_axis_doe_plane[-1], y_axis_doe_plane[-1], y_axis_doe_plane[0]]))
axs2[1,2].set_title("phase_doe_lens_discretized")
axs2[1,2].set_xlabel("[µm]")
axs2[1,2].set_ylabel("[µm]")
divider = make_axes_locatable(axs2[1,2])
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(fig212, cax=cax)

plt.tight_layout()

# 8<------------------------------- param file --------------------------------------

params = ["wavelength [nm]", "d1 [cm]", "d2 [cm]", "doe side length [µm]", "doe diameter [px]", 
          "doe pixel pitch [µm]", "cross diameter [mm]"]
elts = [str(wavelength*1e9), str(100*d1), str(100*d2), str(doe_length*1e6), 
        str(doe_size[0]), str(np.round(pp_doe_plane*1e6, decimals=1)), str(np.round(target_size*pp_image_plane*1e3, decimals=2))]

with open(dir_results+'params.txt', 'w') as f:
    f.write("\n\n")
    for k in range(len(params)):        
            f.write(params[k] + " : " + elts[k] + "\n\n")
            
            





