# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 13:51:43 2024

@author: f24lerou
"""

import numpy as np
import matplotlib.pyplot as plt

#%%
####################### Functions definitions ##########################################

def generateCircularPupil(diameter, **kargs):

#8<---------------------------------------------------------------------------------------------
# generateCircularPupil : generate a circular pupil over a square support
#
# Author : Francois Leroux
# Contact : francois.leroux.pro@gmail.com
# Status : in progress
# Last update : 2023.04.25, Porto
# Comments : for even support resolution, coordinates are defined like [-2,-1,0,1] (N = 4)
#
# Inputs : MANDATORY : diameter {integer}[pixel]
#          OPTIONAL : supportResolution : resolution of the support {tupple (1x2)}[pixel] - default value : diameter
#                     centerCartesianCoordinates : cartesian coordinates {tupple (1x2)}[pixel] - default value : [0,0]
#                     zerosPaddingFactor : magnification factor to be applied to square support in order to do zeros-padding - default value : 1 (no zeros padding)
# Outputs : pupil : circular pupil over square support
#8<---------------------------------------------------------------------------------------------

    # read optinal parameters values
    supportResolution = kargs.get("supportResolution", [diameter,diameter])
    centerCoordinates = kargs.get("centerCoordinates", [0,0])
    zerosPaddingFactor = kargs.get("zerosPaddingFactor", 1)
    
    # for parity issues
    adaptatorX = 1
    adaptatorY = 1
    if supportResolution[0]%2 == 0:
        adaptatorX = 0
    if supportResolution[1]%2 == 0:
        adaptatorY = 0
    
    # define coordinates
    x = np.arange(-(supportResolution[0]//2), supportResolution[0]//2+adaptatorX)
    y = np.flip(np.arange(-(supportResolution[1]//2), supportResolution[1]//2+adaptatorY))
    x = x - centerCoordinates[0]
    y = y - centerCoordinates[1]
    [X,Y] = np.meshgrid(x,y)
    radialCoordinates = np.sqrt(X**2 + Y**2)
    
    # build circular pupil
    pupil = radialCoordinates
    pupil[pupil<=diameter//2] = 1
    pupil[pupil>diameter//2] = 0
    
    # make zeros-padding
    if zerosPaddingFactor>1:
        zerosPaddedSupport = np.zeros([zerosPaddingFactor*supportResolution[0], zerosPaddingFactor*supportResolution[1]])
        zerosPaddedSupport[:supportResolution[0], :supportResolution[1]] = pupil
        pupil = zerosPaddedSupport
    
    return pupil

#%% Main program

# Physical dimensions

pupilDiameter = 225e-6 # diameter of the pupil [m]
wavelength = 1e-6 # wavelength [m]
propagationDistance = 0.5 # [m] # distance between pupil and image plane

# Is Franhoffer approximation valid ?

zMin = 2*np.pi/wavelength * 0.5 * pupilDiameter**2/2

# Sampling parameters

nPx = 32 # number of pixels in the diameter of the pupil

# consequences on the sampling of the frequency space 

    # ???

# consequences on the sampling and dimensions of the image plane

imagePlaneDiameter =  wavelength * propagationDistance * nPx / pupilDiameter # [m]
imagePlaneSamplingStep = imagePlaneDiameter/nPx # [m]

# Simulate the field in pupil and image plane

pupilPlaneField = generateCircularPupil(diameter = nPx, zerosPaddingFactor = 1) 
imagePlaneField = 1/(1j * wavelength * propagationDistance) * np.fft.fftshift(np.fft.fft2(pupilPlaneField))

# Propagate back the field in image plane to the pupil plane

pupilPlaneFieldBackpropagated = 1/(1j * wavelength * propagationDistance) * np.fft.fftshift(np.fft.fft2(imagePlaneField))

# Check energy conservation

totalPowerPupilPlane = np.sum(np.abs(pupilPlaneField)**2) * (pupilDiameter/nPx)**2
totalPowerImagePlane = np.sum(np.abs(imagePlaneField)**2) * (wavelength * propagationDistance / pupilDiameter)**2

totalPowerPupilPlaneBackpropagated = np.sum(np.abs(pupilPlaneFieldBackpropagated)**2) * (pupilDiameter/nPx)**2

# print results

print("\ntotalPowerPupilPlane "+str(totalPowerPupilPlane))
print("totalPowerImagePlane "+str(totalPowerImagePlane))
print("totalPowerPupilPlaneBackpropagated "+str(totalPowerPupilPlaneBackpropagated)+"\n")

#%% test ifft(fft(x)) = x - Works

x = generateCircularPupil(nPx)
xRetrieved = np.fft.ifft2(np.fft.fft2(x))

xTotalPower = np.sum(abs(x)**2)
xRetrievedTotalPower = np.sum(abs(xRetrieved)**2)

absoluteError = xRetrievedTotalPower-xTotalPower
relativeError = absoluteError/xTotalPower

print("\nxTotalPower "+str(xTotalPower))
print("xRetrievedTotalPower "+str(xRetrievedTotalPower))

print("\nabsoluteError "+str(absoluteError))
print("relativeError "+str(relativeError))






