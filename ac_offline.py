"""
Implementación de codificación aritmética OFFLINE específica para piecewise linear predictor.
A diferencia de la versión online, las probabilidades de cada contexto se calculan
usando TODOS los datos antes de comenzar la codificación (distribución fija).

Creada por Lariza Sandoval, Abril 2026.
Versión offline añadida para comparación con versión online.
"""

from collections import defaultdict, Counter
import numpy as np


def arithmetic_encode_offline(symbol_stream: list, raw_stream: list, n_pixels: int, n_classes: int) -> dict:
    PRECISION = 32
    WHOLE     = 1 << PRECISION
    HALF      = WHOLE >> 1
    QUARTER   = WHOLE >> 2

    low     = 0
    high    = WHOLE - 1
    pending = 0
    output  = []

    # --- PASO 1: calcular distribuciones offline con todos los datos ---
    # Contamos cuántos 0s y 1s hay por contexto usando todo el symbol_stream
    counts_offline = defaultdict(lambda: [1, 1])  # Laplace smoothing igual que online
    for ctx, position, bit in symbol_stream:
        counts_offline[ctx][bit] += 1

    # Las probabilidades quedan FIJAS para toda la codificación
    probs = {}
    for ctx, counts in counts_offline.items():
        total = counts[0] + counts[1]
        probs[ctx] = (counts[0], total)  # (n_ceros, total)

    def emit_bit(bit):
        nonlocal pending
        output.append(bit)
        while pending > 0:
            output.append(1 - bit)
            pending -= 1

    def encode_bit_offline(bit, ctx):
        nonlocal low, high, pending

        # Usamos probabilidades FIJAS, no actualizamos
        if ctx in probs:
            n_ceros, total = probs[ctx]
        else:
            n_ceros, total = 1, 2  # fallback Laplace

        mid = low + ((high - low + 1) * n_ceros) // total - 1

        if bit == 0:
            high = mid
        else:
            low = mid + 1

        # NO actualizamos probs aquí — esa es la diferencia clave

        while True:
            if high < HALF:
                emit_bit(0)
                low  = low  * 2
                high = high * 2 + 1
            elif low >= HALF:
                emit_bit(1)
                low  = (low  - HALF) * 2
                high = (high - HALF) * 2 + 1
            elif low >= QUARTER and high < 3 * QUARTER:
                pending += 1
                low  = (low  - QUARTER) * 2
                high = (high - QUARTER) * 2 + 1
            else:
                break

    # --- PASO 2: codificar con probabilidades fijas ---
    for ctx, position, bit in symbol_stream:
        encode_bit_offline(bit, ctx)

    # Flush
    pending += 1
    if low < QUARTER:
        emit_bit(0)
    else:
        emit_bit(1)

    # Raw stream — igual que versión online, fixed-length
    bits_per_raw = int(np.ceil(np.log2(n_classes)))
    raw_output   = []

    for val in raw_stream:
        for i in range(bits_per_raw - 1, -1, -1):
            raw_output.append((val >> i) & 1)

    total_bits = len(output) + len(raw_output)
    bpp_total  = total_bits / n_pixels

    return {
        'bitstream'     : output,
        'raw_bitstream' : raw_output,
        'n_bits'        : len(output),
        'n_raw_bits'    : len(raw_output),
        'bpp_total'     : bpp_total,
        'bpp_symbol'    : len(output) / n_pixels,
        'bpp_raw'       : len(raw_output) / n_pixels,
        'n_classes'     : n_classes,
        'bits_per_raw'  : bits_per_raw,
    }


def arithmetic_decode_offline(encoded: dict, symbol_order: list, n_raw: int, probs_fixed: dict):
    """
    Decodificador offline — requiere las mismas probabilidades fijas usadas en la codificación.
    probs_fixed: dict {ctx: (n_ceros, total)} igual que el encoder genera internamente.
    """
    PRECISION = 32
    WHOLE     = 1 << PRECISION
    HALF      = WHOLE >> 1
    QUARTER   = WHOLE >> 2

    bits   = encoded['bitstream']
    n_bits = len(bits)
    low    = 0
    high   = WHOLE - 1
    value  = 0
    pos    = 0

    # Inicializar value con los primeros PRECISION bits
    for i in range(PRECISION):
        value = (value << 1) | (bits[pos] if pos < n_bits else 0)
        pos  += 1

    symbol_stream = []

    for ctx, position in symbol_order:
        if ctx in probs_fixed:
            n_ceros, total = probs_fixed[ctx]
        else:
            n_ceros, total = 1, 2

        mid = low + ((high - low + 1) * n_ceros) // total - 1

        if value <= mid:
            bit  = 0
            high = mid
        else:
            bit  = 1
            low  = mid + 1

        # NO se actualiza — probabilidades fijas
        symbol_stream.append((ctx, position, bit))

        while True:
            if high < HALF:
                low   = low  * 2
                high  = high * 2 + 1
                value = (value * 2) | (bits[pos] if pos < n_bits else 0)
                pos  += 1
            elif low >= HALF:
                low   = (low  - HALF) * 2
                high  = (high - HALF) * 2 + 1
                value = ((value - HALF) * 2) | (bits[pos] if pos < n_bits else 0)
                pos  += 1
            elif low >= QUARTER and high < 3 * QUARTER:
                low   = (low  - QUARTER) * 2
                high  = (high - QUARTER) * 2 + 1
                value = ((value - QUARTER) * 2) | (bits[pos] if pos < n_bits else 0)
                pos  += 1
            else:
                break

    # Decodificar raw stream
    raw_stream   = []
    raw_bits     = encoded['raw_bitstream']
    bits_per_raw = encoded['bits_per_raw']
    rpos         = 0

    for _ in range(n_raw):
        val = 0
        for i in range(bits_per_raw):
            val = (val << 1) | (raw_bits[rpos] if rpos < len(raw_bits) else 0)
            rpos += 1
        raw_stream.append(val)

    return symbol_stream, raw_stream