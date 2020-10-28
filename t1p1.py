import scipy.io
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
import cv2

save_figures = True
show_figures = True

# Importar imagen, normalizar y obtener tamaño.
path = 'fotos/img.mat'
img = scipy.io.loadmat(path)['img'].astype('uint32')
img = img/np.min(img)
width, height = img.shape

# Crear filtro detector de interseccioens a partir del patrón original
inf = width//2 - width//12
sup = width//2 + width//12 +2
filtro = img[inf:sup, inf:sup]
filtro_norm = (filtro- np.mean(filtro))/(np.linalg.norm(filtro))

# Realizar la convolución
conv = signal.convolve2d(img,filtro_norm)

# Mostrar imagen original, filtro y filtrada
fig1=plt.figure(figsize=[6.4,3])
fig1.suptitle('Detector de intersecciones')
fig1.add_subplot(1,3,1)
plt.title('Original')
plt.imshow(img)
plt.axis('off')
fig1.add_subplot(1,3,2)
plt.title('Filtro')
plt.imshow(filtro_norm)
plt.axis('off')
fig1.add_subplot(1,3,3)
plt.title('Imagen filtrada')
plt.imshow(conv)
plt.axis('off')

# Opcional: Guardar figuras
if save_figures:
    plt.savefig('p1_detector')


# Crear filtro inhibidor de intersecciones
filtro_2 = -filtro_norm

# Realizar la convolución
conv2 = signal.convolve2d(img,filtro_2)

# Mostrar imagen original, filtro y filtrada
fig2=plt.figure(figsize=[6.4,3])
fig2.suptitle('Inhibidor de intersecciones')
fig2.add_subplot(1,3,1)
plt.title('Original')
plt.imshow(img)
plt.axis('off')
fig2.add_subplot(1,3,2)
plt.title('Filtro')
plt.imshow(filtro_2)
plt.axis('off')
fig2.add_subplot(1,3,3)
plt.title('Imagen filtrada')
plt.imshow(conv2)
plt.axis('off')

# Opcional: Guardar figuras
if save_figures:
    plt.savefig('p1_inhibidor')


# Probar filtro con distintos tamaños
# Definir tamaños a probar y crear lista donde se irán guardando las resultantes
original_size = len(filtro_norm)
sizes = [int(x*original_size) for x in [0.25,0.375, 0.5, 0.75, 1]]
filter_size_comparison = []
filter_center_contour = np.zeros((original_size, original_size))
contour = original_size//4
filter_center_contour[contour:-contour,contour:-contour] += 1

# Hacer convoluciones
for size in sizes:
    filtro_i = cv2.resize(filter_center_contour,
                          dsize=(size, size),
                          interpolation=cv2.INTER_LINEAR)
    filter_size_comparison.append(signal.convolve2d(img,filtro_i))

# Mostrar comparación de filtrados
fig3 = plt.figure(figsize=[8,4])
fig3.suptitle('Comparación de tamaños de filtros')
fig3.add_subplot(2,(len(sizes)+1)//2,1)
plt.title('Filtro original')
plt.imshow(filter_center_contour)
plt.axis('off')

for i,size in enumerate(sizes):
    fig3.add_subplot(2,(len(sizes)+1)//2,i+2)
    plt.title('%dx%d' %(size,size))
    plt.imshow(filter_size_comparison[i])
    plt.axis('off')
# Opcional: Guardar imagen
if save_figures:
    plt.savefig('p1_filter_size_comparison')

# Opcional: Mostrar figuras
if show_figures:
    plt.show()
