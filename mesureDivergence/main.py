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
filename_darks_z1 = dirc + r"data\z1_3_20240314_17h20min.avi"
filename_darks_z2 = dirc + r"data\z2_3_20240314_17h17min.avi"
filename_frames_z1 = dirc + r"data\z1_3_20240314_17h20min.avi"
filename_frames_z2 = dirc + r"data\z2_3_20240314_17h17min.avi"
dir_results = dirc + r"results\\"

#%% 8<-------------------------------------- Import modules -----------------------------------

import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit

from mesureDivergence.tools import getFrames, circAverage, Gaussian

#%% 8<-------------------------------------- parameters ----------------------------------------

camera_pp = 0.005/613 # [m] camera pixel pitch (uEye UI-1240LE-M-GL) migth account for magnification of the objective 
                      # when measuring with a screen
delta_z = 0.01        # [m] z2 - z1 : distance between the two measurements
binning = 1           # binning parameter for circular average computation

#%% 8<----------------------------------------- main -------------------------------------------

# get darks from .avi file
darks_array_z1 = getFrames(filename_darks_z1)
darks_array_z2 = getFrames(filename_darks_z2)

# get frames from .avi file
frames_array_z1 = getFrames(filename_frames_z1)
frames_array_z2 = getFrames(filename_frames_z2)

# check the data is not saturated
max_frames_array_z1 = np.max(frames_array_z1)
max_frames_array_z2 = np.max(frames_array_z2)

if max_frames_array_z1 >= 255 or max_frames_array_z2 >= 255:
    print("WARNING : Data is saturated")

# get average dark
average_dark_z1 = np.sum(darks_array_z1, axis=0)
average_dark_z2 = np.sum(darks_array_z2, axis=0)

# get average frame
average_frame_z1 = np.sum(frames_array_z1, axis=0)
average_frame_z2 = np.sum(frames_array_z2, axis=0)

# substract the dark and force negatives values to 0
average_frame_z1 = average_frame_z1 - average_dark_z1
average_frame_z1[average_frame_z1<0] = 0.0
average_frame_z2 = average_frame_z2 - average_dark_z2
average_frame_z2[average_frame_z2<0] = 0.0

# normalization in energy (sum = 1) (superflu)
average_frame_z1 = average_frame_z1/np.sum(average_frame_z1)
average_frame_z2 = average_frame_z2/np.sum(average_frame_z2)

# compute circular average
circular_average_z1, origin_z1 = circAverage(average_frame_z1, binning=binning)
circular_average_z2, origin_z2 = circAverage(average_frame_z2, binning=binning)

# Keep only relevant part
circular_average_z1 = circular_average_z1[:20]
circular_average_z2 = circular_average_z2[:400]

# Normalize max = 1
circular_average_z1 = circular_average_z1/np.max(circular_average_z1)
circular_average_z2 = circular_average_z2/np.max(circular_average_z2)

#%% FWHM method

# find FWHM
fwhm_z1 = 2*np.where(np.abs(circular_average_z1-0.5) ==                          # [m] full width at half maximum
                     np.min(np.abs(circular_average_z1-0.5)))[0][0] * camera_pp
fwhm_z2 = 2*np.where(np.abs(circular_average_z2-0.5) == 
                     np.min(np.abs(circular_average_z2-0.5)))[0][0] * camera_pp  # [m] full width at half maximum

# compute waist (half width at 1/e^2 of the max irradiance) from FWHM
waist_z1 = fwhm_z1/(2*np.log(2))**0.5 # [m]
waist_z2 = fwhm_z2/(2*np.log(2))**0.5 # [m]

# compute divergence (full angle) from FWHM
divergence_fwhm = 2*np.arctan((waist_z2-waist_z1)/delta_z) # [rad]
divergence_fwhm = 180/np.pi * divergence_fwhm                   # [deg]

#%% Gaussian fitting method

# get gaussian fit std
parameters_z1, _ = curve_fit(Gaussian, np.arange(len(circular_average_z1)), circular_average_z1)
fit_sigma_z1 = parameters_z1[0]
parameters_z2, _ = curve_fit(Gaussian, np.arange(len(circular_average_z2)), circular_average_z2)
fit_sigma_z2 = parameters_z2[0]

# compute waist from std
waist_fit_z1 = 2*fit_sigma_z1*camera_pp
waist_fit_z2 = 2*fit_sigma_z2*camera_pp

# compute divergence (full angle) from waist in z1 and z_2
divergence_fit = 2*np.arctan((waist_fit_z2-waist_fit_z1)/delta_z) # [rad]
divergence_fit = 180/np.pi * divergence_fit                       # [deg]

# compute fitted values
fit_z1 = Gaussian(np.arange(len(circular_average_z1)), fit_sigma_z1)
fit_z2 = Gaussian(np.arange(len(circular_average_z2)), fit_sigma_z2)

#%% 8<-------------------------------------- results -------------------------------------------

print("\ndivergence_fwhm : " + str(np.round(divergence_fwhm, decimals=2)) + " °")
print("divergence_fit : " + str(np.round(divergence_fit, decimals=2)) + " °")

#%% 8<--------------------------------------- plots --------------------------------------------

fig, axs = plt.subplots(nrows=2, ncols=2)
axs[0,0].plot(circular_average_z1, 'b')
axs[0,0].plot(circular_average_z2,'r')
axs[0,1].plot(fit_z1, 'b')
axs[0,1].plot(fit_z2, 'r')
axs[1,0].plot(circular_average_z1, '+b')
axs[1,0].plot(fit_z1, 'b')
axs[1,1].plot(circular_average_z2, '+r')
axs[1,1].plot(fit_z2, 'r')



