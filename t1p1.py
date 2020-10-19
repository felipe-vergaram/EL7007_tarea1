# -*- coding: utf-8 -*-
"""
Created on Sun Oct 18 10:58:12 2020

@author: Juan Vergara
"""
import scipy.io
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt

img = scipy.io.loadmat('fotos\img.mat')['img'].astype('uint32')
img = img/np.min(img)

width, height = img.shape

#El patrón se repite 6 veces en ancho y alto
inf = width//2 -width//12 
sup = width//2 +width//12 +2
filtro = img[inf:sup, inf:sup]

filtro_norm = filtro/np.linalg.norm(filtro)
#plt.imshow(filtro)
conv = signal.convolve2d(img,filtro_norm)

fig1=plt.figure()
fig1.add_subplot(1,3,1)
plt.imshow(img)
fig1.add_subplot(1,3,2)
plt.imshow(filtro_norm)
fig1.add_subplot(1,3,3)
plt.imshow(conv)

min_val = np.min(filtro_norm)
max_val = np.max(filtro_norm)
fact = 1+2*min_val/(max_val-min_val)
filtro_2 = -filtro_norm*fact + max_val*fact - min_val

conv2 = signal.convolve2d(img,filtro_2)

fig2=plt.figure()
fig2.add_subplot(1,3,1)
plt.imshow(img)
fig2.add_subplot(1,3,2)
plt.imshow(filtro_2)
fig2.add_subplot(1,3,3)
plt.imshow(conv2)

plt.show()
#fig, ax = plt.subplots(1,2)
#ax[0,0].imshow(img)
#ax[0,1].imshow(filtro)
#plt.imshow(conv)
#plt.show()