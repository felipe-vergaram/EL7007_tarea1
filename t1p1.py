import scipy.io
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt

save_figures = False
show_figures = True

# Importar imagen, normalizar y obtener tamaño. Si se ejecuta en Win10, usar
# el path comentado
#path = 'fotos\\img.mat'
path = 'fotos/img.mat'
img = scipy.io.loadmat(path)['img'].astype('uint32')
img = img/np.min(img)
width, height = img.shape

# Crear filtro detector de interseccioens a partir del patrón original
inf = width//2 - width//12
sup = width//2 + width//12 +2
filtro = img[inf:sup, inf:sup]
filtro_norm = filtro/np.linalg.norm(filtro)

# Realizar la convolución
conv = signal.convolve2d(img,filtro_norm)

# Mostrar imagen original, filtro y filtrada
fig1=plt.figure()
fig1.suptitle('Detector de intersecciones')
fig1.add_subplot(1,3,1)
plt.title('Original')
plt.imshow(img)
fig1.add_subplot(1,3,2)
plt.title('Filtro')
plt.imshow(filtro_norm)
fig1.add_subplot(1,3,3)
plt.title('Imagen filtrada')
plt.imshow(conv)

# Opcional: Guardar figuras
if save_figures:
    plt.savefig('p1_detector')


# Crear filtro inhibidor de intersecciones
min_val = np.min(filtro_norm)
max_val = np.max(filtro_norm)
fact = 1 + 2*min_val/(max_val-min_val)
filtro_2 = -filtro_norm*fact + max_val*fact - min_val

# Realizar la convolución
conv2 = signal.convolve2d(img,filtro_2)

# Mostrar imagen original, filtro y filtrada
fig2=plt.figure()
fig2.suptitle('Inhibidor de intersecciones')
fig2.add_subplot(1,3,1)
plt.title('Original')
plt.imshow(img)
fig2.add_subplot(1,3,2)
plt.title('Filtro')
plt.imshow(filtro_2)
fig2.add_subplot(1,3,3)
plt.title('Imagen filtrada')
plt.imshow(conv2)

# Opcional: Guardar figuras
if save_figures:
    plt.savefig('p1_inhibidor')

# Opcional: Mostrar figuras
if show_figures:
    plt.show()
