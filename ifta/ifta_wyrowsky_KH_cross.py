######################################### IFTA ESSAI KH  ###############################

import numpy as np
import matplotlib.pyplot as plt

from paterns import cross

from phaseMasks import lens

from tools import computeFocal

############################# Directories and filenames #################################

dirc = r"D:\francoisLeroux\codes\ifta\\"
paternFilename = dirc + r"patern\outputs\\"+"cross_5_32x32.npy"
doeFilename = dirc + r"results\\"

######################## Target image definition #######################################

sizeCross = 256
width = 50

############# Read target image and define size and position ############################

target_image = cross(sizeCross, width=width) # read target image
targSizeX, targSizeY = target_image.shape    # size in X and Y of target image
doeSizeX = 512                               # X size of DOE 
doeSizeY = 512                               # Y size of DOE
offsetX = doeSizeX//2 - targSizeX//2         # X position of the top-left corner of the target image 
offsetY = doeSizeY//2 - targSizeY//2         # Y position of the top-left corner of the target image 

############ Define target amplitude float array and initiate input field
target_amp = np.asarray(target_image, float) # conversion target image to float
target_amp = np.sqrt(target_amp)             # Target image amplitude

amp_image = np.zeros((doeSizeX, doeSizeY))                                    # Amplitude output field = 0
amp_image[offsetX:offsetX+targSizeX,offsetY:offsetY+targSizeY] = target_amp   # Amplitude = target image in window
phase_image = 2*np.pi*np.random.rand(doeSizeX, doeSizeY)                      # Random image phase


###############################  1st IFTA loop - image diffuser ##################
max_iter = 25 # Number of iterations
rfact = 1.2 # Reinforcement factor: strengthens reconstructed image. Typical: 1.2
champs_image = amp_image*np.exp(1j * phase_image) # Initiate input field

for iter in range(max_iter):
    champs_image=np.fft.ifftshift(champs_image)
    champs_DOE = np.fft.ifft2(champs_image) # DOE = TF-1 image
    phase_DOE = np.angle(champs_DOE) # on récupère la phase du DOE
    champs_DOE = np.exp(phase_DOE * 1j) # remplacement de l'amplitude par 1 (celle du laser)
    champs_image = np.fft.fft2(champs_DOE) # image = TF du DOE
    champs_image = np.fft.fftshift(champs_image)
    phase_image = np.angle(champs_image) # on recupere la phase de l'image
    amp_image[offsetX:offsetX+targSizeX,offsetY:offsetY+targSizeY]=rfact*target_amp
    champs_image = amp_image*np.exp(phase_image * 1j) # remplacement de l'amplitude par celle de l'image et phase récupérée

pm1=phase_DOE
recovery1 = np.absolute(np.fft.fftshift(np.fft.fft2(np.exp(pm1 * 1j))))**2 # transformee de Fourier pour avoir l'image finale reconstruite

############################### 2nd IFTA loop - DOE quantification ##################
# NOT gradual quantification !!! But for N>4 should be OK

N=8 # Number of DOE phase levels

rfact = 1.2 # Reinforcement factor: strengthens reconstructed image. typical: 1.2
max_iter2=40
champs_image = np.fft.fftshift(np.fft.fft2(np.exp(phase_DOE * 1j)))
phase_image = np.angle(champs_image)

for iter in range(max_iter2):
    champs_image=np.fft.ifftshift(champs_image)
    champs_DOE = np.fft.ifft2(champs_image) # DOE = TF-1 image
    #champs_DOE=np.fft.ifftshift(champs_DOE)
    phase_DOE = np.angle(champs_DOE) # on recupere la phase du DOE
    phase_DOE = (np.pi/(N/2))*np.round(phase_DOE*((N/2)/np.pi)) # phase quantification
    champs_DOE = np.exp(phase_DOE * 1j) # remplacement de l'amplitude par 1 (celle du laser)
    champs_image = np.fft.fft2(champs_DOE) # image = TF du DOE
    champs_image = np.fft.fftshift(champs_image)
    phase_image = np.angle(champs_image) # on recupEre la phase de l'image
    amp_image[offsetX:offsetX+targSizeX,offsetY:offsetY+targSizeY]=rfact*target_amp
    champs_image = amp_image*np.exp(phase_image * 1j) # remplacement de l'amplitude par celle de l'image et phase recuperee


###################### Calculate, display and plot the output plane and quantified DOE phase

recovery2 = np.absolute(np.fft.fftshift(np.fft.fft2(np.exp(phase_DOE * 1j))))**2 # transformee de Fourier pour avoir l'image finale reconstruite
pm2 = (N*(np.pi + phase_DOE)/(2.0*np.pi))%N    # Change range from -Pi...Pi to 0...N-1

fig, axs = plt.subplots(nrows=1, ncols=2)
axs[0].imshow(pm2)
axs[0].set_title("DOE - phase ("+str(N)+" levels)")
axs[1].imshow(recovery2)
axs[1].set_title("Image plane - Irradiance")
plt.savefig(dirc+r"figures\\"+"maFigure.png")

plt.imsave(doeFilename+'pm2.png', pm2)
plt.imsave(doeFilename+'recovery2.png', recovery2)

efficiency_loop1=np.sum(recovery1[offsetX:offsetX+targSizeX,offsetY:offsetY+targSizeY])/np.sum(recovery1)*100
efficiency_loop2=np.sum(recovery2[offsetX:offsetX+targSizeX,offsetY:offsetY+targSizeY])/np.sum(recovery2)*100
print("DE after 1st loop = %2.2f" % efficiency_loop1)
print("DE after 2nd loop = %2.2f" % efficiency_loop2)

