# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 10:19:24 2024

@author: f24lerou
"""

# 8<---------------------------- Add path --------------------------------

import os
import sys

path = os.path.abspath(os.path.abspath('..')) # path = 'D:\\francoisLeroux\\codes'
sys.path.append(path)

#%% 8<------------------------------ Directories and filenames --------------------------------

dirc = os.path.abspath(os.getcwd()) + r"\\" # dirc = 'D:\\francoisLeroux\\codes\\mesureDivergence\\\\'
filename = dirc + r"data\test.avi"
dir_results = dirc + r"results\\"

#%% 8<-------------------------------------- Import modules -----------------------------------

import cv2
import numpy as np

#%% 8<--------------------------------------- main -------------------------------------------

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

# Print the shape of the frames array
print("Shape of frames array:", frames_array.shape)

# Save the frames array as a .npy file
np.save(dir_results+'test.npy', frames_array)
