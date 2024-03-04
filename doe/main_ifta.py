# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 10:07:22 2024

@author: f24lerou
"""

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

n_levels = 3

d1 = 0.01               # [m] distance laser - DOE
d2 = 0.05               # [m] distance DOE - image plane
doe_length = 225e-6     # [m] length of the side of the DOE

wavelength = 1e-6       # [m] wavelength

# 8<--------------------- main ------------------------------------------

target = cross(sizeCross=sizeCross, width=width)

phase_DOE, recovery, efficiency = ifta(target, doe_size, n_levels=n_levels, compute_efficiency=1)

f = computeFocal(d1, d2) # focal length for source - image plane conjugation
phase_lens = lens(f, wavelength=wavelength, sizeSupport=doe_size, samplingStep=doe_length/doe_size[0], n_levels=n_levels)

phase_DOE_lens = discretization(phase_DOE+phase_lens, n_levels)

# 8<-------------------- results ----------------------------------------

print("efficiency = "+str(efficiency))

fig, axs = plt.subplots(nrows=2, ncols=2)
fig1=axs[0,0].imshow(phase_DOE)
axs[0,0].set_title("phase_DOE ("+str(n_levels)+" levels)")
plt.colorbar(fig1, ax=axs[0,0])
fig2=axs[0,1].imshow(recovery)
axs[0,1].set_title("Image plane - Irradiance")
plt.colorbar(fig2, ax=axs[0,1])
fig3=axs[1,0].imshow(phase_lens)
axs[1,0].set_title("phase_lens ("+str(n_levels)+" levels)")
plt.colorbar(fig3, ax=axs[1,0])
fig4=axs[1,1].imshow(phase_DOE_lens)
axs[1,1].set_title("phase_DOE_lens ("+str(n_levels)+" levels)")
plt.colorbar(fig4, ax=axs[1,1])

plt.tight_layout()
plt.savefig(dirc+r"figures\\"+"test.png")





