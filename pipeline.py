"""
Created on 2026-04-10
Created by: Lariza Sandoval
Piecewise Pixel Prediction — Sección II.A
Xie & Klimesh, IPN Progress Report 42-169, 2007

Funciones:
    entropy                — calcula la entropía de una distribución de valores
    get_neighbors          — obtiene p1, p2, p3, p4 para un píxel
    encode_pixel           — genera los bits de decisión para un píxel
    decode_pixel           — reconstruye el píxel a partir de los bits de decisión
    encode_image           — aplica la predicción a toda la imagen → residuos
    decode_image           — aplica la predicción inversa → imagen original
    pattern_based_context   — asigna un contexto basado en el patrón de vecinos y posición de la pregunta
    encode_to_entropy_input — genera el flujo de símbolos para codificación entropica con modelo de contexto
    entropy_pattern_context — calcula la entropía total usando el modelo de contexto
    pipeline                — función principal que ejecuta todo el pipeline y retorna métricas
    context_model_analysis  — analiza la distribución de decisiones por contexto
    residual_stats          — calcula estadísticas sobre los residuos para análisis

"""

import numpy as np
from aritmetic import arithmetic_encode, arithmetic_decode
from ac_offline import arithmetic_encode_offline
from collections import defaultdict
#import pandas as pd


def entropy(data):
    values, counts = np.unique(data, return_counts=True)
    probs = counts / counts.sum()
    return -np.sum(probs * np.log2(probs))

# ---------------------------------------------------------------------------
# Neighborhood definition and utilities
# ---------------------------------------------------------------------------

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
        'positions' : list[int]  — posiciones de las preguntas realizadas
        'raw'       : int | None — valor crudo de x si no hubo coincidencia
    """
    bits      = []
    positions = []
    seen      = set()
    pos       = 0

    for pk in (p1, p3, p2, p4):
        if pk in seen:
            pos += 1
            continue
        seen.add(pk)
        if x == pk:
            bits.append(0)
            positions.append(pos)
            return {'bits': bits, 'positions': positions,
                    'raw': None, 'matched_pk': pk}

        bits.append(1)
        positions.append(pos)
        pos += 1

    return {'bits': bits, 'positions': positions,
            'raw': x, 'matched_pk': None}

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
    residuals[row][col] = {'bits': [...], 'raw': int|None}
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
# Prediccion + modelo de contexto para codificación entropica 
# ---------------------------------------------------------------------------

def pattern_based_context(p1: int, p2: int, p3: int, p4: int, position: int) -> int:
    """
    Retorna un entero entre 0 y 36 que identifica el contexto.
    Posiciones: 0=primera pregunta p1, 1=segunda p3, 2=tercera p2, 3=cuarta p4
    """
    PATTERN_CONTEXT_MAP = {
    ('AAAA', 0): 0,
    ('AAAB', 0): 1,  ('AAAB', 3): 2,
    ('AABA', 0): 3,  ('AABA', 1): 4,
    ('AABB', 0): 5,  ('AABB', 1): 6,
    ('ABAA', 0): 7,  ('ABAA', 2): 8,
    ('ABAB', 0): 9,  ('ABAB', 2): 10,
    ('ABAC', 0): 11, ('ABAC', 2): 12, ('ABAC', 3): 13,
    ('ABBA', 0): 14, ('ABBA', 1): 15,
    ('ABBB', 0): 16, ('ABBB', 1): 17,
    ('ABBC', 0): 18, ('ABBC', 1): 19, ('ABBC', 3): 20,
    ('ABCA', 0): 21, ('ABCA', 1): 22, ('ABCA', 2): 23,
    ('ABCB', 0): 24, ('ABCB', 1): 25, ('ABCB', 2): 26,
    ('ABCC', 0): 27, ('ABCC', 1): 28, ('ABCC', 2): 29,
    ('ABCD', 0): 30, ('ABCD', 1): 31, ('ABCD', 2): 32, ('ABCD', 3): 33,
    ('AABC', 0): 34, ('AABC', 1): 35, ('AABC', 3): 36,
    }

    # Calcular el patrón
    labels = {}
    next_label = 0
    pattern = []
    for pk in (p1, p2, p3, p4):
        if pk not in labels:
            labels[pk] = next_label
            next_label += 1
        pattern.append(labels[pk])

    letter_map = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
    pattern_name = ''.join(letter_map[l] for l in pattern)
    #print(f"pattern: {pattern_name}, position: {position}") 
    key = (pattern_name, position)
    if key not in PATTERN_CONTEXT_MAP:
        raise ValueError(f"Combinación inválida: patrón={pattern_name}, posición={position}")

    return PATTERN_CONTEXT_MAP[key]

def encode_to_entropy_input(image: np.ndarray, context_model: str = 'pattern'):
    """
    Realiza la prediccion de la imagen y aplica un modelo de contexto para cada bit de decisión.
        Retorna dos listas:
            symbol_stream: lista de tuplas (contexto, posición, bit)
            raw_stream: lista de valores crudos para píxeles sin coincidencia
        El modelo de contexto puede ser 'pattern' o 'edge_based' (no implementado).
        Para cada bit de decisión, se determina el contexto según el patrón de vecinos y la posición de la pregunta.
    """
    H, W = image.shape
    symbol_stream = []
    raw_stream    = []

    for row in range(H):
        for col in range(W):
            x = int(image[row, col])
            p1, p2, p3, p4 = get_neighbors(image, row, col)
            residual = encode_pixel(x, p1, p2, p3, p4)
            # Usar posición real, no índice del bit
            for bit, position in zip(residual['bits'], residual['positions']):
                if context_model == 'pattern':
                    ctx = pattern_based_context(p1, p2, p3, p4, position)
                else:
                    ctx = 0 #edge_based_context(image, row, col)

                symbol_stream.append((ctx, position, bit))
            if residual['raw'] is not None:
                raw_stream.append(residual['raw'])
    return symbol_stream, raw_stream

def entropy_pattern_context(symbol_stream, raw_stream, n_pixels):
    """
    Calcula la entropía total en bits por píxel usando
    el context model pattern-based.

    Parámetros
    ----------
    symbol_stream : lista de tuplas (ctx, position, bit)
    raw_stream    : lista de valores crudos
    n_pixels      : total de píxeles de la imagen
    
    """
    from collections import defaultdict

    # Tabla de contadores por contexto
    tabla = defaultdict(lambda: [0, 0])
    for ctx, position, bit in symbol_stream:
        tabla[ctx][bit] += 1
        #tabla[(ctx, position)][bit] += 1
    #print(f"Tabla: {dict(tabla)}")
    # Entropía condicional de los bits de decisión

    # Primera pasada — contar total de bits de decisión
    n_decision_bits = sum(c[0] + c[1] for c in tabla.values())

    # Segunda pasada — calcular entropía condicional
    H_conditional = 0.0

    total_bits   = 0
    total_H_bits = 0.0
    #[print(f"Contexto {ctx}: {counts}") for ctx, counts in tabla.items()]
    for ctx, counts in tabla.items():
        ceros, unos = counts
        total = ceros + unos
        p_ctx = total / n_decision_bits  # p(c)
        if total == 0:
            continue
        p0 = ceros / total
        p1 = unos  / total
        H_ctx = 0.0
        if p0 > 0: H_ctx -= p0 * np.log2(p0)
        if p1 > 0: H_ctx -= p1 * np.log2(p1)
        H_conditional += p_ctx * H_ctx  # [bits/bit de decisión]
        #print(f"entropia + peso contexto {ctx}: {total} * {H_ctx} = {total * H_ctx} bits/decision (p0={p0:.4f}, p1={p1:.4f})")
        total_H_bits +=  total * H_ctx
        total_bits   += total
    #bpp_bits = total_H_bits / n_pixels

    # Entropía de los valores crudos

    if raw_stream:
        #print("Esto es una prueba",np.unique(raw_stream) )
        bpp_raws = (entropy(raw_stream) * len(raw_stream))/ n_pixels 
        #print(f"Valoreeeess crudos {raw_stream}")
    else:
        bpp_raws = 0.0
    
    # Expected compressed data rate [bits/pixel]
    R_decision = H_conditional * (n_decision_bits / n_pixels)

    #print(f"Entropía bits decisión (con contexto): {bpp_bits} bits/pixel")
    #print(f"Entropía valores crudos: {bpp_raws} bits/pixel")
    return { 
        "entropy_bits_decision": H_conditional, #H_conditional,
        "bpp_bits": R_decision ,#bpp_bits,
        "bpp_raws": bpp_raws,   
        "total_bpp": R_decision + bpp_raws 
        }

#---------------------------------------------------------------------------


#---------------------------------------------------------------------------
def pipeline(img):

    #seq = entropy_seq_context(img, encode_pixel, get_neighbors) tratar los bit en grupo 110, 0, 1110, etc
    symbol_stream, raw_stream = encode_to_entropy_input(img)
    encoded = arithmetic_encode(symbol_stream, raw_stream, img.size, img.max() + 1)
    enconded_offline = arithmetic_encode_offline(symbol_stream, raw_stream, img.size, img.max() + 1)
    encoded['n_pixels'] = int(img.max()) + 1
    entropy_info_offline = entropy_pattern_context(symbol_stream, raw_stream, img.size)
    decision_bits = [bit for _, _, bit in symbol_stream]
   
    residuals = encode_image(img)
    bits_pixel = []
    for row in residuals:
        for res in row:
            joint_bits = "".join(map(str, res['bits']))
            bits_pixel.append(joint_bits)
    #print("bits pixel sample:", bits_pixel)
    #print(f"Entropy: {entropy(bits_pixel)}")

    return {
        'n_clases': img.max() + 1,
        'H_img': entropy(img), #Entropía de la imagen original en bits / pixel
        'H_seq': entropy(bits_pixel), # bits / pixel 
        #'H_seq_cxt': seq['H_seq_ctx'], # bits / símbolo de secuencia con contexto
        #'Bits_per_pixel_seq': seq['bpp_bits_seq'], # bits / pixel esperado con secuencia completa sin contexto
        #'Exp_bit_rate_seq_cxt': entropy_seq['bpp_seq_cxt'], # bits / pixel esperado con secuencia completa y contexto
        'H_bits': entropy(decision_bits), # entropía de los bits de decisión bit / bit de decisión 
        #'H_bits_cxt_online': entropy_info['entropy_bits_decision'], # bits / bit de decisión con contexto
        'H_bits_cxt': entropy_info_offline['entropy_bits_decision'], # bits / bit de decisión con contexto offline
        'Exp_bit_rate_only_bits_seq': entropy(decision_bits) * len(decision_bits) / img.size, # bits / pixel solo por los bits de decisión sin contexto
        'Exp_bit_rate' : entropy_info_offline['total_bpp'], # bits / pixel total esperado con contexto
        'Exp_bit_rate_bits' : entropy_info_offline['bpp_bits'], # bits / pixel solo por los bits de decisión con contexto
        'Exp_bit_rate_raw' : entropy_info_offline['bpp_raws'], # bits / pixel solo por los valores crudos
        'bit_rate_final'  : encoded['bpp_total'], # bits / pixel tOTAL
        'bit_rate_offline' : enconded_offline['bpp_total'], # bits / pixel TOTAL con AC offline
        'bit_rate_bits'   : encoded['bpp_symbol'], # bits / pixel para los bits de decisión codificados con modelo de contexto
        'bit_rate_raw'      : encoded['bpp_raw'], # bits / pixel para los valores crudos codificados
       }

def context_model_analysis(img):
    symbol_stream, raw_stream = encode_to_entropy_input(img)
    tabla = defaultdict(lambda: [0, 0])
    for ctx, position, bit in symbol_stream:
        tabla[ctx][bit] += 1
    diccionario_columnas = {
        "n_clases": img.max() + 1,
        "contexto": list(tabla.keys()),
        "counts_0": [conteos[0] for conteos in tabla.values()],
        "counts_1": [conteos[1] for conteos in tabla.values()],
        "total_decisions": [sum(conteos) for conteos in tabla.values()],
    }
    return diccionario_columnas

# ---------------------------------------------------------------------------
# Estadísticas de los residuos
# ---------------------------------------------------------------------------

def residual_stats(img):
    """
    Calcula estadísticas sobre los residuos para análisis.
    Retorna un dict con:
        'n_pixels'     : total de píxeles
        'n_decisions'  : distribución de cuántos bits se emitieron (1..4)
        'n_unmatched'  : píxeles sin coincidencia (valor crudo)
        'pct_1bit'     : porcentaje resuelto con 1 sola decisión
        'pct_unmatched': porcentaje sin coincidencia
    """
    residuals = encode_image(img)
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
        'n_clases':None,
        'n_pixels': total,
        #'1_decisions': counts.get(1, 0),
        #'2_decisions': counts.get(2, 0),
        #'3_decisions': counts.get(3, 0),
        #'4_decisions': counts.get(4, 0),
        #'n_unmatched': unmatched,
        'pct_1bit': 100 * counts.get(1, 0) / total if total else 0,
        'pct_2bits': 100 * counts.get(2, 0) / total if total else 0,
        'pct_3bits': 100 * counts.get(3, 0) / total if total else 0,
        'pct_4bits': 100 * counts.get(4, 0) / total if total else 0,
        'pct_unmatched': 100 * unmatched / total if total else 0,
    }
