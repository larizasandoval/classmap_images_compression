import numpy as np
from sklearn.cluster import KMeans

def cargar_imagen_raw(ruta_archivo, bandas=224, lineas=512, columnas=614, dtype=np.uint16):
    """
    Carga una imagen multibanda en formato RAW y la organiza en una matriz indexada (Bandas, Líneas, Columnas).
    """
    datos_puros = np.fromfile(ruta_archivo, dtype=dtype)
    imagen_3d = datos_puros.reshape((bandas, lineas, columnas))
    return imagen_3d

def clasificar_con_kmeans(imagen_3d, n_clusteres=4):
    """
    Aplica el algoritmo K-Means para agrupar las firmas espectrales de la imagen sin supervisión.
    
    Parámetros:
    - imagen_3d: Matriz de dimensiones (bandas, lineas, columnas).
    - n_clusteres: Número de clases que queremos identificar (ej. Water, Land, Forest, Human Habitat).
    """
    bandas, lineas, columnas = imagen_3d.shape
    
    # 1. Reorganizar la matriz: (bandas, lineas, columnas) -> (lineas * columnas, bandas)
    # Cada fila es un vector de características (firma espectral de 224 componentes)
    imagen_espectral = imagen_3d.reshape(bandas, -1).T
    
    # 2. Inicializar y entrenar K-Means
    print(f"Entrenando K-Means para encontrar {n_clusteres} clases espectrales...")
    # 'mini_batch' o n_init='auto' ayudan a la eficiencia según la versión de sklearn
    kmeans = KMeans(n_clusters=n_clusteres, init='k-means++', random_state=42, n_init=10)
    
    # El modelo encuentra los centroides y asigna una etiqueta a cada píxel
    predicciones = kmeans.fit_predict(imagen_espectral)
    
    # 3. Devolver el mapa de clases a su forma espacial original (líneas, columnas)
    mapa_clases = predicciones.reshape((lineas, columnas))
    
    return mapa_clases

# --- EJEMPLO DE USO ---
if __name__ == "__main__":
    # Dimensiones basadas en las escenas del artículo
    BANDAS = 224
    LINEAS = 512
    COLUMNAS = 614
    
    ruta_raw = "curated_images/f970620t01p02_r03_sc01.c-s16be-224x512x614.raw"
    
    try:
        # Carga del volumen de datos hiperespectrales
        img_hiper = cargar_imagen_raw(ruta_raw, bandas=BANDAS, lineas=LINEAS, columnas=COLUMNAS)
        
        # Definimos el número de clases (por ejemplo, 4 clases como el mapa conceptual del paper)
        N_CLASSES = 15 
        
        # Ejecutar la segmentación espectral automática
        mapa_de_clases_kmeans = clasificar_con_kmeans(img_hiper, n_clusteres=N_CLASSES)
        
        # Guardar el mapa resultante en formato binario RAW (valores del 0 al 3)
        mapa_de_clases_kmeans.astype(np.uint8).tofile(f"curated_images/mapa_kmeans_final_{N_CLASSES}_clases.raw")
        print("¡Proceso completado! El mapa no supervisado por K-Means se guardó exitosamente.")
        
    except FileNotFoundError:
        print(f"No se pudo encontrar el archivo en la ruta: {ruta_raw}")