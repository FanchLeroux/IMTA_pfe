# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 11:21:57 2024

@author: f24lerou
"""

# 8<-------------------------------------------- Add path -------------------------------------

import os
import sys

path = os.path.abspath(os.path.abspath('..'))
sys.path.append(path)

#%% 8<---------------------------------------- Import modules ------------------------------------

import numpy as np
import cv2

from doe.tools import getCartesianCoordinates

#%% 8<------------------------------------ Functions definitions ---------------------------------

def getFrames(filename):
    
    """
    getFrames : read .avi file from uEye UI-1240LE-M-GL camera and return the frames as an 3D np array
    
    Author : Francois Leroux
    Contact : francois.leroux.pro@gmail.com
    Status : in progress
    Last update : 2024.03.13, Brest
    Comments : 
      
    Inputs : MANDATORY : filename : filename of the video to read
                        
    Outputs : frames_array : 3D np array containing the frames. Shape : (nFrames, nPixel, nPixel)
    """
    
    # Define the path to the pre-recorded video file
    video_path = filename  # Change this to your video file path

    # Capture video from file
    cap = cv2.VideoCapture(video_path)

    # Initialize an empty list to store frames
    frames_list = []

    while(cap.isOpened()): # ?
        ret, frame = cap.read() # ?
        if ret==True: # ?
            
            # Append the frame to the frames list
            frames_list.append(frame)
            
            # Press 'q' to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    # Release everything if job is finished
    cap.release() # ?

    # Convert the frames list to a 3D numpy array
    frames_array = np.array(frames_list)
    frames_array = frames_array[:,:,:,0]     # monochrome but still 3 identical values returned per pixel. Keep only one
        
    return frames_array


def circAverage(img, origin = None, samplingStep = 1, binning = 1):
    
    """
    circAverage : compute circular average over 2D image
    
    Author : Francois Leroux
    Contact : francois.leroux.pro@gmail.com
    Status : in progress
    Last update : 2024.03.13, Brest
    Comments :  librement inspiré de https://levelup.gitconnected.com/a-simple-method-to-calculate-circular-circular_average-
                                     averages-in-images-4186a685af3
                binning : moyenne à droite
                getCartesianCoordinates : For even support size N, coordinates are defined like [-2,-1,0,1] (N = 4)
    
    Inputs : MANDATORY : img {2D numpy Array} : image on which circular average needs to be computed
             OPTIONAL : origin {tupple 1x2} : origin around which the circular average will be computed. In nrows x ncols.
                                              default : origin = none : the circular average is computed around the 
                                              max of img
                        binning : full length of the range of values considered to get one point of the radial average
                                  default : binning = 1
    Output : circular_average {1D numpy Array}
    """
    
    img = np.array(img)
    
    # if not specified, setting origin to the maximum of the image
    if origin == None:
        origin = np.where(img == img.max())
        
        # in case the max appear multiple times
        if len(origin[0])>1:
            origin = (origin[0][0], origin[1][0])
            print("WARNING! The maximum appear multiple times. The first occurence is choosed as the origin for \
            circular average computation")
    
    # compute coordinates
    [X,Y] = getCartesianCoordinates(nrows=img.shape[0], ncols=img.shape[1])
    origin = [X[origin[0], origin[1]], Y[origin[0], origin[1]]] # origin in cartesian coordinates
    X = X - origin[0] # offset to place the (0,0) point at the origin
    Y = Y - origin[1] # offset to place the (0,0) point at the origin
    radial_coordinates = (X**2 + Y **2)**0.5 # [px] radial coordinates, center = origin

    rad = np.arange(0, np.max(radial_coordinates), 1) # abscisse
    
    circular_average = np.zeros(len(rad))
    index = 0
    
    for i in rad:
        mask = (np.greater_equal(radial_coordinates, i) & 
                np.less_equal(radial_coordinates, i + binning)) # take into account values between i and i + binning  
        values = img[mask]
        circular_average[index] = np.mean(values)
        index += 1
    
    return circular_average, origin

def Gaussian(x, sigma):
    y = np.exp(-1*x**2/(2*sigma**2))
    return y




