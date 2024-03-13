# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 10:19:24 2024

@author: f24lerou
"""

# 8<-------------------------------------- Add path ------------------------------------------

import os
import sys

path = os.path.abspath(os.path.abspath('..')) # path = 'D:\\francoisLeroux\\codes'
sys.path.append(path)

#%% 8<------------------------------ Directories and filenames --------------------------------

dirc = os.path.abspath(os.getcwd()) + r"\\"   # dirc = 'D:\\francoisLeroux\\codes\\mesureDivergence\\\\'
filename_z1 = dirc + r"data\test.avi"
filename_z2 = dirc + r"data\test.avi"
dir_results = dirc + r"results\\"

#%% 8<-------------------------------------- Import modules -----------------------------------

import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit

from mesureDivergence.tools import getFrames, circAverage

#%% 8<--------------------------------------- main -------------------------------------------

# get frames from .avi file
frames_array_z1 = getFrames(filename_z1)
frames_array_z2 = getFrames(filename_z2)

# get average frame
average_frame_z1 = np.sum(frames_array_z1, axis=0)
average_frame_z2 = np.sum(frames_array_z1, axis=0)

# compute circular average
circular_average_z1 = circAverage(average_frame_z1)
circular_average_z2 = circAverage(average_frame_z2)

# Gaussian fitting as in https://education.molssi.org/python-data-analysis/03-data-fitting/index.html


#%% 8<--------------------------------------- plots --------------------------------------------

fig, axs = plt.subplots(nrows=1, ncols=2)
axs[0].plot(circular_average_z1)
axs[1].plot(circular_average_z2)