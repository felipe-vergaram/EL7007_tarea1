# -*- coding: utf-8 -*-
"""
Created on Sun Oct 18 10:58:12 2020

@author: Juan Vergara
"""
import scipy
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt

img = scipy.io.loadmat('img.mat')['img'].astype('uint32')
img = img/np.min(img)

width, height = img.shape

#El patrón se repite 6 veces en ancho y alto
inf = width//2 -width//12 
sup = width//2 +width//12 +2
filtro = img[inf:sup, inf:sup]

filtro_norm = filtro/np.linalg.norm(filtro)
#plt.imshow(filtro)
conv = signal.convolve2d(img,filtro_norm)

fig=plt.figure()
fig.add_subplot(1,3,1)
plt.imshow(img)
fig.add_subplot(1,3,2)
plt.imshow(filtro_norm)
fig.add_subplot(1,3,3)
plt.imshow(conv)
#fig, ax = plt.subplots(1,2)
#ax[0,0].imshow(img)
#ax[0,1].imshow(filtro)
#plt.imshow(conv)
#plt.show()