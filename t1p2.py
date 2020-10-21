import scipy.io
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
import imageio


save_figures = True
show_figures = True

# Importar imagen. Si se ejecuta en Windows, usar el path comentado
# path = 'fotos\\faces.jpg' # Windows 10
path = 'fotos/faces.jpg' # Linux
img = imageio.imread(path)

# Definir filtro pasabajos dado el tamaño dxd
d = 30
lowpass_filter = np.ones((d,d))/(d**2)

# Calcular convolución de imagen con filtro pasabajos
lowpass = np.zeros(img.shape)
for channel in range(3):
    lowpass[:,:,channel] = signal.convolve2d(img[:,:,channel],
                                             lowpass_filter,
                                             mode = 'same')
lowpass = np.uint8(lowpass)

# Mostrar imagen original y filtrada
fig = plt.figure(figsize=[6.4,3])
fig.add_subplot(1,2,1)
plt.title('Imagen original')
plt.imshow(img)
plt.axis('off')
fig.add_subplot(1,2,2)
plt.title('Imagen filtrada con pasabajos')
plt.imshow(lowpass)
plt.axis('off')

# Opcional: Guardar imagen
if save_figures:
    plt.savefig('p2_lowpass')


# Comparación de distintos tamaños de filtros
sizes = [5,15,30,60]
lowpass_size_comparison = []
for size in sizes:
    # Definir filtro pasabajos dado el tamaño sizexsize
    lowpass_filter_i = np.ones((size,size))/(size**2)

    # Calcular convolución de imagen con filtro pasabajos
    lowpass_i = np.zeros(img.shape)
    for channel in range(3):
        lowpass_i[:,:,channel] = signal.convolve2d(img[:,:,channel],
                                                 lowpass_filter_i,
                                                 mode = 'same')
    # Agregar a lowpass_size_comparison
    lowpass_size_comparison.append(np.uint8(lowpass_i))

# Mostrar comparación de filtrados
fig2 = plt.figure(figsize=[8,2.5])
fig2.suptitle('Comparación de tamaños de filtros')
for i,size in enumerate(sizes):
    fig2.add_subplot(1,len(sizes),i+1)
    plt.title('%dx%d' %(size,size))
    plt.imshow(lowpass_size_comparison[i])
    plt.axis('off')
# Opcional: Guardar imagen
if save_figures:
    plt.savefig('p2_lowpass_size_comparison')

# Creación de filtro pasabanda a partir de 2 filtros pasabajos auxiliares
# Defnir filtro pasabajos auxiliar 1 de tamaño d1xd1
d1 = 15
aux_lowpass_filter_1 = np.ones((d1,d1))/(d1**2)

# Calcular convolución de imagen con filtro pasabajos auxiliar 1
aux_lowpass_1 = np.zeros(img.shape)
for channel in range(3):
    aux_lowpass_1[:,:,channel] = signal.convolve2d(img[:,:,channel],
                                                   aux_lowpass_filter_1,
                                                   mode = 'same')
aux_lowpass_1 = np.uint8(aux_lowpass_1)

# Defnir filtro pasabajos auxiliar 2 de tamaño d2xd2
d2 = 13
aux_lowpass_filter_2 = np.ones((d2,d2))/(d2**2)

# Calcular convolución de imagen con filtro pasabajos auxiliar 2
aux_lowpass_2 = np.zeros(img.shape)
for channel in range(3):
    aux_lowpass_2[:,:,channel] = signal.convolve2d(img[:,:,channel],
                                                   aux_lowpass_filter_2,
                                                   mode = 'same')
aux_lowpass_2 = np.uint8(aux_lowpass_2)

# Cálculo de imagen filtrada con pasabanda. La división por 4 es solo estética
bandpass = (aux_lowpass_2-aux_lowpass_1)//4

# Mostrar imagen original y filtrada
fig3 = plt.figure(figsize=[6.4,3])
fig3.add_subplot(1,2,1)
plt.title('Imagen original')
plt.imshow(img)
plt.axis('off')
fig3.add_subplot(1,2,2)
plt.title('Imagen filtrada con pasabanda')
plt.imshow(bandpass)
plt.axis('off')

# Opcional: Guardar imagen
if save_figures:
    plt.savefig('p2_bandpass')

# Opcional: Mostrar figuras
if show_figures:
    plt.show()
