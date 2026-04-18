"""
Piecewise Pixel Prediction — Sección II.A
Xie & Klimesh, IPN Progress Report 42-169, 2007

Funciones:
    get_neighbors          — obtiene p1, p2, p3, p4 para un píxel
    encode_pixel           — genera los bits de decisión para un píxel
    decode_pixel           — reconstruye el píxel a partir de los bits de decisión
    encode_image           — aplica la predicción a toda la imagen → residuos
    decode_image           — aplica la predicción inversa → imagen original
"""
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Neighborhood definition and utilities
# ---------------------------------------------------------------------------


def entropy(data):
    values, counts = np.unique(data, return_counts=True)
    probs = counts / counts.sum()
    return -np.sum(probs * np.log2(probs))

def get_neighbors(image: np.ndarray, row: int, col: int):
    """
    Retorna (p1, p2, p3, p4) para el píxel en (row, col).
    Layout (Figura 3 del paper):
        p2  p3  p4
        p1   x

    p3 = norte      (row-1, col)
    p1 = oeste      (row,   col-1)
    p2 = noroeste   (row-1, col-1)
    p4 = noreste    (row-1, col+1)
    Píxeles fuera de borde → valor 0 (convención del paper).
    """
    H, W = image.shape
    def v(r, c):
        return int(image[r, c]) if 0 <= r < H and 0 <= c < W else 0
    return v(row, col-1), v(row-1, col-1), v(row-1, col), v(row-1, col+1)

# ---------------------------------------------------------------------------
# Codificación de un píxel → residuo
# ---------------------------------------------------------------------------

def encode_pixel(x: int, p1: int, p2: int, p3: int, p4: int):
    """
    Genera la representación residual de x a partir de sus vecinos.

    Hace hasta 4 preguntas en orden p1, p3, p2, p4:
        ¿x == pk?  →  bit 0 (sí, parar)  /  bit 1 (no, continuar)
    Las preguntas redundantes (vecino ya preguntado) se saltan.
    Si x no coincide con ningún vecino, se incluye el valor crudo.
    Retorna un dict con:
        'bits'      : list[int]  — secuencia de bits de decisión (0s y 1s)
        'raw'       : int | None — valor crudo de x si no hubo coincidencia
    """
    bits = []
    seen = set()
    for pk in (p1, p3, p2, p4):
        if pk in seen:
            continue
        seen.add(pk)

        if x == pk:
            bits.append(0)
            return {'bits': bits, 'raw': None}
        bits.append(1)

    return {'bits': bits, 'raw': x}

# ---------------------------------------------------------------------------
# Decodificación de un residuo → píxel original
# ---------------------------------------------------------------------------

def decode_pixel(residual: dict, p1: int, p2: int, p3: int, p4: int):
    """
    Reconstruye el valor original de x a partir del residuo.
    Retorna el valor entero del píxel reconstruido.
    """
    bits = residual['bits']
    seen = []

    for pk in (p1, p3, p2, p4):
        if pk in seen:
            continue
        seen.append(pk)

    for i, bit in enumerate(bits):
        if bit == 0:
            return seen[i]

    return residual['raw']


# ---------------------------------------------------------------------------
# Codificación de imagen completa → mapa de residuos
# ---------------------------------------------------------------------------

def encode_image(image: np.ndarray):
    """
    Aplica la predicción píxel a píxel en orden raster.
    Retorna una lista de listas de residuos 
    residuals[row][col] = {'bits': [...], 'raw': int|None, 'matched_pk': int|None}
    """
    H, W = image.shape
    residuals = [[None] * W for _ in range(H)]
    for row in range(H):
        for col in range(W):
            x = int(image[row, col])
            p1, p2, p3, p4 = get_neighbors(image, row, col)
            residuals[row][col] = encode_pixel(x, p1, p2, p3, p4)

    return residuals

# ---------------------------------------------------------------------------
# Decodificación de residuos → imagen reconstruida
# ---------------------------------------------------------------------------

def decode_image(residuals: list, shape: tuple):
    """
    Reconstruye la imagen original a partir de los residuos.
    """
    H, W = shape
    image = np.zeros((H, W), dtype=np.uint8)
    for row in range(H):
        for col in range(W):
            p1, p2, p3, p4 = get_neighbors(image, row, col)
            image[row, col] = decode_pixel(residuals[row][col], p1, p2, p3, p4)

    return image


# ---------------------------------------------------------------------------
# Estadísticas de los residuos
# ---------------------------------------------------------------------------

def residual_stats(residuals: list):
    """
    Calcula estadísticas sobre los residuos para análisis.
    Retorna un dict con:
        'n_pixels'     : total de píxeles
        'n_decisions'  : distribución de cuántos bits se emitieron (1..4)
        'n_unmatched'  : píxeles sin coincidencia (valor crudo)
        'pct_1bit'     : porcentaje resuelto con 1 sola decisión
        'pct_unmatched': porcentaje sin coincidencia
    """
    counts = {1: 0, 2: 0, 3: 0, 4: 0}
    unmatched = 0
    total = 0
    for row in residuals:
        for res in row:
            total += 1
            n = len(res['bits'])
            if res['raw'] is not None:
                unmatched += 1
            else:
                counts[n] = counts.get(n, 0) + 1
    return {
        'n_pixels': total,
        'n_decisions': counts,
        'n_unmatched': unmatched,
        'pct_1bit': 100 * counts.get(1, 0) / total if total else 0,
        'pct_unmatched': 100 * unmatched / total if total else 0,
    }


# ---------------------------------------------------------------------------
# Main — demostracion
# ---------------------------------------------------------------------------

if __name__ == '__main__':

    print("=" * 54)
    print("  Piecewise Pixel Prediction — demo 3 clases")
    print("=" * 54)
    
    #img = np.fromfile("sintetic_map_512x512.raw", dtype=np.uint8).reshape((512, 512))
    #residuals = encode_image(img)
    #img_rec = decode_image(residuals, img.shape)
    #assert np.array_equal(img, img_rec), f"ERROR: fallo en reconstrucción"
    #stats = residual_stats(residuals)
    #print(f"Residual stats:\n {stats}")
    #bits = [bit['bits'] for row in residuals for bit in row]


    # Image to test the encoder/decoder on a small example
    test = np.array([[1, 2, 3,2],
                     [2, 1, 1,3], 
                     [3, 3, 1,1],
                     [1, 1, 1,1]
                     ], dtype=np.uint8)    
    
    test_residuals = encode_image(test)
    img_rec_test = decode_image(test_residuals, test.shape)
    assert np.array_equal(test, img_rec_test), f"ERROR: fallo en reconstrucción"
    print(f"Test image:\n{test}")
    print("Residuals:")
    [print(dicc) for test_res in test_residuals for dicc in test_res]
