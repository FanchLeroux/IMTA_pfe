######################################### IFTA ESSAI KH  ###############################
#########################################               ################################
########################################################################################

import cv2
import numpy as np

############# Read target image and define size and position ############################
filename = 'arrow25x30.pgm'
target_image = cv2.imread(filename, 0)       # read target image
targsizeX, targsizeY = target_image.shape    # size in X and Y of target image
DOEsizeX = 1024                      # X size of target image 
DOEsizeY = 1024                      # Y size of target image 
offsetX = 520                       # X position of target image 
offsetY = 500                       # Y position of target image 

############ Define target amplitude float array and initiate input field
target_amp = np.asarray(target_image, float) # conversion target image to float
target_amp = np.sqrt(target_amp)             # Target image amplitude

amp_image = np.zeros((DOEsizeX, DOEsizeY))                                    # Amplitude output field = 0
amp_image[offsetX:offsetX+targsizeX,offsetY:offsetY+targsizeY]=target_amp     # Amplitude = target image in window
phase_image = 2*np.pi*np.random.rand(DOEsizeX, DOEsizeY)                      # Random image phase


###############################  1st IFTA loop - image diffuser ##################
max_iter = 25 # Number of iterations
rfact = 1.2  # Reinforcement factor: strengthens reconstructed image
champs_image = amp_image*np.exp(phase_image * 1j) # Initiate input field

for iter in range(max_iter):
    champs_image=np.fft.ifftshift(champs_image)
    champs_DOE = np.fft.ifft2(champs_image) # DOE = TF-1 image
    phase_DOE = np.angle(champs_DOE) # on récupère la phase du DOE
    champs_DOE = np.exp(phase_DOE * 1j) # remplacement de l'amplitude par 1 (celle du laser)
    champs_image = np.fft.fft2(champs_DOE) # image = TF du DOE
    champs_image = np.fft.fftshift(champs_image)
    phase_image = np.angle(champs_image) # on recupere la phase de l'image
    amp_image[offsetX:offsetX+targsizeX,offsetY:offsetY+targsizeY]=rfact*target_amp
    champs_image = amp_image*np.exp(phase_image * 1j) # remplacement de l'amplitude par celle de l'image et phase récupérée

pm1=phase_DOE
saturation = 1.0   # Camera Saturation of output: 1.0 unsaturated, < 1.0 underexposed, > 1.0 overexposed
recovery1 = np.absolute(np.fft.fftshift(np.fft.fft2(np.exp(pm1 * 1j))))**2 # transformee de Fourier pour avoir l'image finale reconstruite
recovery1 *= (saturation*255.0/recovery1.max())  # normalise 0-255 for image output

#cv2.imshow('Phase: stage1',pm1)
#display = cv2.convertScaleAbs(recovery1)
#cv2.imshow('Reconstruction: stage1',display)

############################### 2nd IFTA loop - DOE quantification ##################
# NOT gradual qualtification !!! But for N>4 should be OK

N=4 #256 Number of DOE phase levels

rfact = 1.2  # Reinforcement factor: strengthens reconstructed image
max_iter2=40
champs_image = np.fft.fftshift(np.fft.fft2(np.exp(phase_DOE * 1j)))
phase_image = np.angle(champs_image)

for iter in range(max_iter2):
    champs_image=np.fft.ifftshift(champs_image)
    champs_DOE = np.fft.ifft2(champs_image) # DOE = TF-1 image
    #champs_DOE=np.fft.ifftshift(champs_DOE)
    phase_DOE = np.angle(champs_DOE) # on recupere la phase du DOE
    phase_DOE = (np.pi/(N/2))*np.round(phase_DOE*((N/2)/np.pi))
    champs_DOE = np.exp(phase_DOE * 1j) # remplacement de l'amplitude par 1 (celle du laser)
    champs_image = np.fft.fft2(champs_DOE) # image = TF du DOE
    champs_image = np.fft.fftshift(champs_image)
    phase_image = np.angle(champs_image) # on recupEre la phase de l'image
    amp_image[offsetX:offsetX+targsizeX,offsetY:offsetY+targsizeY]=rfact*target_amp
    champs_image = amp_image*np.exp(phase_image * 1j) # remplacement de l'amplitude par celle de l'image et phase recuperee


###################### Calculate, display and plot the output plane and quantified DOE phase

recovery2 = np.absolute(np.fft.fftshift(np.fft.fft2(np.exp(phase_DOE * 1j))))**2 # transformee de Fourier pour avoir l'image finale reconstruite
efficiency=np.sum(recovery2[offsetX:offsetX+targsizeX,offsetY:offsetY+targsizeY])/np.sum(recovery2)*100
saturation = 1.0   # Camera Saturation of output: 1.0 unsaturated, < 1.0 underexposed, > 1.0 overexposed
recovery2 *= (saturation*255.0/recovery2.max())  # normalise 0-255 for image output

pm2 = (256*(np.pi + phase_DOE)/(2.0*np.pi))%256    # Change range from -Pi...Pi to 0...256 
display = cv2.convertScaleAbs(pm2)
cv2.imshow('DOE Phase',display)
cv2.imwrite('DOE-Phase.png', pm2)

display = cv2.convertScaleAbs(recovery2)
cv2.imshow('Reconstruction',display)
cv2.imwrite('reconstruction.png', recovery2)

efficiency_loop1=np.sum(recovery1[offsetX:offsetX+targsizeX,offsetY:offsetY+targsizeY])/np.sum(recovery1)*100
efficiency_loop2=np.sum(recovery2[offsetX:offsetX+targsizeX,offsetY:offsetY+targsizeY])/np.sum(recovery2)*100
print("DE after 1st loop = %2.2f" % efficiency_loop1)
print("DE after 2nd loop = %2.2f" % efficiency_loop2)

cv2.waitKey(0)
cv2.destroyAllWindows()
