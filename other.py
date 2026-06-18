import os
import numpy as np
import jpeg_ls

# --- CONFIGURACIÓN DE TU IMAGEN RAW (1 BANDA) ---
# Cambia estos valores por la resolución real de tu imagen
ANCHO = 512
ALTO = 614
# ------------------------------------------------

# Función para leer el archivo .raw de 1 banda
def leer_imagen_raw_1banda(ruta_archivo, ancho, alto):
    # Lee los bytes del archivo binario
    datos_crudos = np.fromfile(ruta_archivo, dtype=np.uint8)
    
    # Al ser 1 banda, la estructura es simplemente (Alto, Ancho)
    imagen_matriz = datos_crudos.reshape((alto, ancho))
    return imagen_matriz

# --- NOTA SOBRE JPEG-LS ---
# Asegúrate de importar aquí tu módulo real de JPEG-LS.
# El estándar JPEG-LS soporta nativamente imágenes de 1 banda (escala de grises).
# import jpeg_ls
# ---------------------------

# Simulación del módulo jpeg_ls (reemplázalo con tu librería real)
class DummyJpegLs:
    def encode(self, data):
        return data.tobytes()
    def decode(self, buffer):
        return np.frombuffer(buffer, dtype=np.uint8).reshape((ALTO, ANCHO))

#jpeg_ls = DummyJpegLs()
# -----------------------------------------------------------------


# Ruta de tu imagen
fname_img = 'curated_maps/Moffet_Sc02_9classes_PriorityMap-u8be-1x512x614.raw'  # Cambia esto por la ruta real de tu imagen RAW

if not os.path.exists(fname_img):
    print(f"Error: No se encuentra el archivo en la ruta '{fname_img}'")
else:
    # Leemos el RAW de 1 banda
    data_image = leer_imagen_raw_1banda(fname_img, ANCHO, ALTO)

    # Comprimir los datos de la imagen a una secuencia de bytes.
    data_buffer = jpeg_ls.encode(data_image)

    # Tamaños.
    size_raw = os.path.getsize(fname_img)
    
    print('Size of 1-band image data:     {:n}'.format(len(data_image.tobytes())))
    print('Size of RAW original file:     {:n}'.format(size_raw))
    print('Size of JPEG-LS encoded data:  {:n}'.format(len(data_buffer)))
    print(data_buffer[:100])  # Mostrar los primeros 100 bytes del buffer comprimido

    # Descomprimir.
    data_image_b = jpeg_ls.decode(data_buffer)

    # Comparar.
    is_same = (data_image == data_image_b).all()
    print('Restored data is identical to original: {:s}'.format(str(is_same)))