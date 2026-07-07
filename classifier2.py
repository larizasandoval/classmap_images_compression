import numpy as np
from sklearn.svm import SVC
import os

def cargar_imagen_raw(ruta_archivo, bandas=224, lineas=512, columnas=614, dtype=np.uint16):
    """
    Carga una imagen multibanda en formato RAW y la organiza en una matriz indexada.
    Se asume un orden típico de almacenamiento continuo (BSQ o similar).
     Ajusta el 'dtype' (p. ej., np.float32 o np.int16) según tus datos reales.
    """
    # Leer el archivo binario completo
    datos_puros = np.fromfile(ruta_archivo, dtype=dtype)
    
    # Redimensionar a la estructura original (Bandas, Líneas, Columnas)
    imagen_3d = datos_puros.reshape((bandas, lineas, columnas))
    
    return imagen_3d

def entrenar_y_clasificar(imagen_3d, pixeles_entrenamiento, etiquetas):
    """
    Entrena un clasificador SVM con pixeles de muestra y genera el mapa de clases.
    
    Parámetros:
    - imagen_3d: Matriz de dimensiones (bandas, lineas, columnas).
    - pixeles_entrenamiento: Matriz de forma (N, bandas) con las firmas espectrales de muestra.
    - etiquetas: Vector de forma (N,) con el índice de clase correspondiente.
    """
    bandas, lineas, columnas = imagen_3d.shape
    
    # 1. Reorganizar la imagen completa para la clasificación por píxel
    # Pasamos de (bandas, lineas, columnas) -> (lineas * columnas, bandas)
    # Cada fila representará un píxel con sus 224 características (bandas)
    imagen_espectral = imagen_3d.reshape(bandas, -1).T
    
    # 2. Inicializar y entrenar el clasificador SVM
    # Usamos RBF (Radial Basis Function), muy común para teledetección
    print("Entrenando el clasificador SVM...")
    clasificador = SVC(kernel='rbf', C=10.0, gamma='scale', random_state=42)
    clasificador.fit(pixeles_entrenamiento, etiquetas)
    
    # 3. Predecir las clases para todos los píxeles de la escena
    print("Generando el mapa de clasificación...")
    predicciones = clasificador.predict(imagen_espectral)
    
    # 4. Reconstruir la forma espacial de la imagen (líneas, columnas)
    mapa_clases = predicciones.reshape((lineas, columnas))
    
    return mapa_clases

# --- EJEMPLO DE USO ---
if __name__ == "__main__":
    # Configuración de dimensiones basándonos en el paper
    BANDAS = 224
    LINEAS = 512
    COLUMNAS = 614
    
    # Reemplaza con la ruta de tu archivo raw
    ruta_raw = "tu_imagen_aviris.raw" 
    path = "curated_images"
    archivos = [f"{path}/{f}" for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]

    
    try:
        # Cargar la imagen hiperespectral
        img_hiper = cargar_imagen_raw(ruta_raw, bandas=BANDAS, lineas=LINEAS, columnas=COLUMNAS)
        
        # --- Simulación de datos de entrenamiento ---
        # En un escenario real, debes extraer estos píxeles de posiciones conocidas (coordenadas x, y)
        # Por ejemplo: muestra_agua = img_hiper[:, y_agua, x_agua]
        
        N_MUESTRAS = 100  # Número de píxeles para entrenar
        firmas_ejemplo = np.random.rand(N_MUESTRAS, BANDAS)  # Datos ficticios de entrenamiento
        etiquetas_ejemplo = np.random.randint(0, 4, size=N_MUESTRAS)  # 4 clases (p.ej. Agua, Tierra...)
        
        # Clasificar la escena completa
        mapa_de_clases = entrenar_y_clasificar(img_hiper, firmas_ejemplo, etiquetas_ejemplo)
        
        # Guardar el resultado en formato binario o procesarlo posteriormente
        mapa_de_clases.astype(np.uint8).tofile("mapa_clasificacion_final.raw")
        print("¡Proceso completado! El mapa de clases se guardó exitosamente.")
        
    except FileNotFoundError:
        print(f"No se encontró el archivo en la ruta especificada: {ruta_raw}")