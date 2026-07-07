import os
import re
import numpy as np
from sklearn.cluster import KMeans


def cargar_imagen_dinamica(ruta_archivo):
    """Carga una imagen multibanda analizando las dimensiones y el tipo de dato

    desde el formato al final del nombre (ej: ...u16be-72x1225x406.raw)
    """
    nombre_completo = os.path.basename(ruta_archivo)
    nombre_sin_ext = os.path.splitext(nombre_completo)[0]

    match = re.search(r"(\w+)-(\d+)x(\d+)x(\d+)$", nombre_sin_ext)
    if not match:
        return None

    tipo_dato, z_str, y_str, x_str = match.groups()

    bandas = int(z_str)
    lineas = int(y_str)
    columnas = int(x_str)

    if "u16be" in tipo_dato:
        dt = np.dtype(">u2")
    elif "u16le" in tipo_dato:
        dt = np.dtype("<u2")
    elif "i16be" in tipo_dato:
        dt = np.dtype(">i2")
    elif "i16le" in tipo_dato:
        dt = np.dtype("<i2")
    elif "u8" in tipo_dato:
        dt = np.dtype("u1")
    elif "f32be" in tipo_dato:
        dt = np.dtype(">f4")
    else:
        dt = np.dtype(np.uint16)

    datos_puros = np.fromfile(ruta_archivo, dtype=dt)
    imagen_3d = datos_puros.reshape((bandas, lineas, columnas))

    return imagen_3d, bandas, lineas, columnas, tipo_dato


def construir_vectores_con_contexto_espacial(
    imagen_3d, bandas_seleccionadas=None, n_bandas=32, ventana=3
):
    """Construye vectores de características incluyendo el vecindario 3x3."""
    bandas, lineas, columnas = imagen_3d.shape

    if bandas_seleccionadas is None:
        bandas_seleccionadas = np.linspace(0, bandas - 1, n_bandas, dtype=int)

    img_sub = imagen_3d[bandas_seleccionadas, :, :]
    pad_width = ventana // 2

    img_padded = np.pad(
        img_sub,
        pad_width=((0, 0), (pad_width, pad_width), (pad_width, pad_width)),
        mode="reflect",
    )

    vectores_caracteristicas = []
    for i in range(ventana):
        for j in range(ventana):
            vecino = img_padded[:, i : i + lineas, j : j + columnas]
            vecino_aplanado = vecino.reshape((n_bandas, -1)).T
            vectores_caracteristicas.append(vecino_aplanado)

    return np.hstack(vectores_caracteristicas)


def clasificar_con_kmeans_contextual(
    imagen_3d, n_clusteres=9, n_bandas=32, ventana=3
):
    """Ejecuta el clustering K-Means."""
    _, lineas, columnas = imagen_3d.shape
    X_contextual = construir_vectores_con_contexto_espacial(
        imagen_3d, n_bandas=n_bandas, ventana=ventana
    )

    kmeans = KMeans(
        n_clusters=n_clusteres, init="k-means++", random_state=42, n_init=10
    )
    predicciones = kmeans.fit_predict(X_contextual)
    return predicciones.reshape((lineas, columnas))


# --- PROCESAMIENTO OPTIMIZADO Y CONTROLADO ---
if __name__ == "__main__":
    path_raiz = "images_complete/green_book_corpus"

    # =========================================================================
    # CONFIGURACIÓN DE CONTROL DE CÓMPUTO
    # =========================================================================
    # Opción A: Procesar todo -> CARPETAS_A_PROCESAR = None
    # Opción B: Procesar solo sensores específicos -> CARPETAS_A_PROCESAR = ['landsat', 'aviris']
    # Pon aquí los nombres exactos de las subcarpetas que quieres procesar en esta tanda.
    CARPETAS_A_PROCESAR = ['msg']#None

    # Granularidades de clases (Xie & Klimesh)
    N_CLASSES = [4, 7, 9, 17, 32]
    # =========================================================================

    if not os.path.exists(path_raiz):
        print(f"La carpeta raíz '{path_raiz}' no existe.")
        exit()

    print(f"Iniciando escaneo de imágenes en: {path_raiz}")

    for raiz_actual, subcarpetas, archivos in os.walk(path_raiz):
        # Saltarse las carpetas de mapas resultantes
        if "maps" in raiz_actual:
            continue

        # Obtener el nombre de la carpeta del sensor actual
        nombre_sensor = os.path.basename(raiz_actual)

        # FILTRO 1: Si definiste carpetas específicas y la actual no está en la lista, la saltamos
        if (
            CARPETAS_A_PROCESAR is not None
            and nombre_sensor not in CARPETAS_A_PROCESAR
        ):
            continue

        for archivo in archivos:
            ruta_completa_archivo = os.path.join(raiz_actual, archivo)

            # Intentar verificar si es una imagen válida mediante el nombre
            nombre_base_archivo = os.path.splitext(archivo)[0]
            match_verificacion = re.search(
                r"(\w+)-(\d+)x(\d+)x(\d+)$", nombre_base_archivo
            )

            if not match_verificacion:
                continue

            # Crear ruta de la carpeta de mapas para este sensor
            carpeta_destino_maps = os.path.join(raiz_actual, "maps")
            os.makedirs(carpeta_destino_maps, exist_ok=True)

            # FILTRO 2: Comprobar qué mapas de esta imagen YA fueron generados antes
            clases_pendientes = []
            for n_clases in N_CLASSES:
                nombre_salida_esperado = os.path.join(
                    carpeta_destino_maps,
                    f"mapa_{nombre_base_archivo}_kmeans_{n_clases}clases.raw",
                )
                if not os.path.exists(nombre_salida_esperado):
                    clases_pendientes.append(n_clases)

            # Si ya se generaron todos los mapas para este archivo, nos lo saltamos por completo
            if not clases_pendientes:
                print(
                    f"  [Saltado] {archivo} ya tiene todos sus mapas generados."
                )
                continue

            # --- CARGA ASÍNCRONA/DEMANDADA (Solo lee si hace falta calcular algo) ---
            resultado_carga = cargar_imagen_dinamica(ruta_completa_archivo)
            if resultado_carga is None:
                continue

            img_hiper, bandas, lineas, columnas, tipo_dato = resultado_carga

            print("-" * 70)
            print(f"Sensor: [{nombre_sensor}] | Archivo: {archivo}")
            print(
                f"  -> Ejecutando clases pendientes: {clases_pendientes} de {N_CLASSES}"
            )

            try:
                N_BANDAS_SUB = min(32, bandas)

                for n_clases in clases_pendientes:
                    mapa_contextual = clasificar_con_kmeans_contextual(
                        img_hiper,
                        n_clusteres=n_clases,
                        n_bandas=N_BANDAS_SUB,
                        ventana=3,
                    )

                    nombre_salida = os.path.join(
                        carpeta_destino_maps,
                        f"mapa_{re.sub(r'-[us]16(be|le)-\d+x\d+x\d+$', '', nombre_base_archivo)}_{n_clases}clases-u8be-1x{lineas}x{columnas}_.raw",
                    )
                    mapa_contextual.astype(np.uint8).tofile(nombre_salida)
                    print(f"    [OK] Creado: K-Means {n_clases} clases")

            except Exception as e:
                print(
                    f"  [Error] Falló el procesamiento del archivo {archivo}: {e}"
                )

    print("\n" + "=" * 70)
    print("¡Tanda de procesamiento finalizada!")
    print("=" * 70)