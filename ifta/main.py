# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 10:07:22 2024

@author: f24lerou
"""

import numpy as np
import matplotlib.pyplot as plt

from paterns import cross

from phaseMasks import lens

from tools import computeFocal

from ifta import ifta

# 8<---------------- Directories and filenames ---------------------------

dirc = r"D:\francoisLeroux\codes\ifta\\"
paternFilename = dirc + r"patern\outputs\\"+"cross_5_32x32.npy"
doeFilename = dirc + r"results\\"

# 8<--------------------- Parameters ------------------------------------

# target image
sizeCross = 50
width = 5

doe_size = [512, 512]

n_levels = 8

# 8<--------------------- main ------------------------------------------

target = cross(sizeCross=sizeCross, width=width)

phase_DOE, recovery, efficiency = ifta(target, doe_size, n_levels=n_levels, compute_efficiency=1)



# 8<-------------------- results ----------------------------------------

print("efficiency = "+str(efficiency))

fig, axs = plt.subplots(nrows=1, ncols=2)
axs[0].imshow(phase_DOE)
axs[0].set_title("DOE - phase ("+str(n_levels)+" levels)")
axs[1].imshow(recovery)
axs[1].set_title("Image plane - Irradiance")
plt.savefig(dirc+r"figures\\"+"test.png")





