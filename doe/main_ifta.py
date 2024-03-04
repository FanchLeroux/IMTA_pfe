# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 10:07:22 2024 under Python 3.11.7

@author: f24lerou
"""

from mpl_toolkits.axes_grid1 import make_axes_locatable

import numpy as np
import matplotlib.pyplot as plt

from doe.paterns import cross

from doe.phaseMasks import lens

from doe.tools import computeFocal, discretization

from doe.ifta import ifta

# 8<---------------- Directories and filenames ---------------------------

dirc = r"D:\francoisLeroux\codes\doe\\"
paternFilename = dirc + r"patern\outputs\\"+"cross_5_32x32.npy"
doeFilename = dirc + r"results\\"

# 8<--------------------- Parameters ------------------------------------

# target image
sizeCross = 50
width = 5

doe_size = [512, 512]

n_levels = 4

d1 = 0.01               # [m] distance laser - DOE
d2 = 0.05               # [m] distance DOE - image plane
doe_length = 225e-6     # [m] length of the side of the DOE

wavelength = 1e-6       # [m] wavelength

# 8<--------------------- main ------------------------------------------

target = cross(sizeCross=sizeCross, width=width, sizeSupport = [sizeCross+10,sizeCross+10])

phase_doe, recovery, efficiency = ifta(target, doe_size, n_levels=n_levels, compute_efficiency=1)

f = computeFocal(d1, d2) # focal length for source - image plane conjugation
phase_lens = lens(f, wavelength=wavelength, sizeSupport=doe_size, samplingStep=doe_length/doe_size[0], n_levels=0)

phase_doe_lens = phase_doe + phase_lens
phase_doe_lens_discretized = discretization(phase_doe+phase_lens, n_levels)

# 8<-------------------- results ----------------------------------------

np.save(doeFilename+"crossDoe", phase_doe)
np.save(doeFilename+"crossDoeLens", phase_doe_lens)


print("efficiency = "+str(efficiency))

fig, axs = plt.subplots(nrows=2, ncols=3)

fig00=axs[0,0].imshow(target, aspect="equal")
axs[0,0].set_title("target")

fig01=axs[0,1].imshow(phase_doe)
axs[0,1].set_title("phase_doe ("+str(n_levels)+" levels)")
divider = make_axes_locatable(axs[0,1])
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(fig01, cax=cax)

fig02=axs[0,2].imshow(recovery)
axs[0,2].set_title("Image plane - Irradiance\n efficiency = "+str(round(efficiency*100))+"%")
divider = make_axes_locatable(axs[0,2])
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(fig02, cax=cax)

fig10=axs[1,0].imshow(phase_lens)
axs[1,0].set_title("phase_lens")
divider = make_axes_locatable(axs[1,0])
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(fig10, cax=cax)

fig11=axs[1,1].imshow(phase_doe_lens)
axs[1,1].set_title("phase_doe_lens ("+str(n_levels)+" levels)")
divider = make_axes_locatable(axs[1,1])
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(fig11, cax=cax)

fig12=axs[1,2].imshow(phase_doe_lens_discretized)
axs[1,2].set_title("phase_doe_lens_discretized")
divider = make_axes_locatable(axs[1,2])
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(fig12, cax=cax)

plt.tight_layout()
plt.savefig(dirc+r"figures\\"+"test.png")







