# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 16:19:09 2024 under Python 3.11.7

@author: f24lerou
"""

# 8<---------------------------- Add path --------------------------------

import os
import sys

path = os.path.abspath(os.path.abspath('..'))
sys.path.append(path)

# 8<--------------------------- Import modules ---------------------------

import numpy as np
import matplotlib.pyplot as plt

from doe.tools import computeFocal, discretization
from doe.phaseMasks import lens, tilt

# 8<---------------- Directories and filenames ---------------------------

dirc = r"D:\francoisLeroux\codes\metalenses\\"
pmFilename = dirc + r"results\\"

pm_size = [512, 512]

n_levels = 4

d1 = 0.01              # [m] distance laser - pm
d2 = 0.03              # [m] distance pm - image plane
pm_length = 500e-6     # [m] length of the side of the DOE

height = 0.001          # [m] height of the image point with respect to the optical axis


wavelength = 1e-6      # [m] wavelength

deltaPhi = 2*np.pi/wavelength * height/d2 * pm_length   # [rad] phase difference between the two edges of the phase screen
pm_tilt = tilt(deltaPhi=deltaPhi, sizeSupport=pm_size)

samplingStep=pm_length/pm_size[0]

f = computeFocal(d1, d2)
pm_lens = lens(f, wavelength=wavelength, sizeSupport=pm_size, samplingStep=samplingStep)

pm = pm_tilt + pm_lens
pm_discretized = discretization(pm, n_levels=n_levels)

fig, axs = plt.subplots(nrows=1, ncols=4)

fig0=axs[0].imshow(pm_lens)
axs[0].set_title("pm_lens")

fig1=axs[1].imshow(pm_tilt)
axs[1].set_title("pm_tilt")

fig2=axs[2].imshow(pm)
axs[2].set_title("pm")

fig3=axs[3].imshow(pm_discretized)
axs[3].set_title("pm_discretized")



