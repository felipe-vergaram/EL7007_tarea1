import scipy.io
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
import imageio
from scipy import ndimage

img = imageio.imread('fotos/faces.jpg')
#img = imageio.imread('fotos\\faces.jpg')
conv = np.zeros(img.shape)
d=30
filtro = np.ones((d,d))/(d**2)
for channel in range(3):
    conv[:,:,channel] = signal.convolve2d(img[:,:,channel], filtro,mode='same')

#conv = signal.convolve(img,filtro,'same').astype(uint(8))
conv=np.uint8(conv)
fig=plt.figure()
fig.add_subplot(1,2,1)
plt.imshow(img)
fig.add_subplot(1,2,2)
plt.imshow(conv)



conv2 = np.zeros(img.shape)
d2=15
filtro2 = np.ones((d2,d2))/(d2**2)
for channel in range(3):
    conv2[:,:,channel] = signal.convolve2d(img[:,:,channel], filtro2,mode='same')
conv2=np.uint8(conv2)

conv3 = np.zeros(img.shape)
d3=13
filtro3 = np.ones((d3,d3))/(d3**2)
for channel in range(3):
    conv3[:,:,channel] = signal.convolve2d(img[:,:,channel], filtro3,mode='same')
conv3=np.uint8(conv3)

conv4= (conv3-conv2)//4


fig2=plt.figure()
fig2.add_subplot(1,4,1)
plt.imshow(img)
fig2.add_subplot(1,4,2)
plt.imshow(conv2)
plt.title('d2 = %d' %(d2))
#plt.imshow(ndimage.gaussian_filter(img, f1))
fig2.add_subplot(1,4,3)
plt.imshow(conv3)
plt.title('d3 = %d' %(d3))
#plt.imshow(ndimage.gaussian_filter(img, f2))
fig2.add_subplot(1,4,4)
plt.imshow(conv4)
#plt.imshow(gauss_bandpass)
plt.savefig('test/00_2_13_3')
plt.close(fig2)
#plt.show()
