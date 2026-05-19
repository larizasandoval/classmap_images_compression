"""
Implementación de codificación aritmética especifica para picewise linear predictor. 
Basada en el algoritmo clásico de codificación aritmética, 
pero adaptada para manejar dos flujos de bits: uno para los símbolos (decisiones) y otro para los valores crudos (residuos).

Creada por Lariza Sandoval, Abril 2026.

"""



from collections import defaultdict 
import numpy as np
from fractions import Fraction
from collections import Counter



def arithmetic_encode(symbol_stream: list, raw_stream: list, n_pixels: int,n_classes: int) -> dict:
    PRECISION = 32
    WHOLE     = 1 << PRECISION
    HALF      = WHOLE >> 1
    QUARTER   = WHOLE >> 2

    low     = 0
    high    = WHOLE - 1
    pending = 0
    output  = []

    counts = defaultdict(lambda: [1, 1])

    def emit_bit(bit):
        nonlocal pending
        output.append(bit)
        while pending > 0:
            output.append(1 - bit)
            pending -= 1

    def encode_bit(bit, ctx):
        nonlocal low, high, pending

        c     = counts[ctx]
        total = c[0] + c[1]
        # mid es el punto de division entre 0 y 1
        mid   = low + ((high - low + 1) * c[0]) // total - 1

        if bit == 0:
            high = mid
        else:
            low = mid + 1

        counts[ctx][bit] += 1

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

    for ctx, position, bit in symbol_stream:
        encode_bit(bit, ctx)

    # Flush
    pending += 1
    if low < QUARTER:
        emit_bit(0)
    else:
        emit_bit(1)

    # Raw stream con bits fijos
    #n_classes    = int(max(raw_stream)) + 1 if raw_stream else 2
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


def arithmetic_decode(encoded: dict, symbol_order: list, n_raw: int):
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

    counts = defaultdict(lambda: [1, 1])

    # Inicializar value con los primeros PRECISION bits
    for i in range(PRECISION):
        value = (value << 1) | (bits[pos] if pos < n_bits else 0)
        pos  += 1

    symbol_stream = []

    for ctx, position in symbol_order:
        c     = counts[ctx]
        total = c[0] + c[1]
        mid   = low + ((high - low + 1) * c[0]) // total - 1

        if value <= mid:
            bit  = 0
            high = mid
        else:
            bit = 1
            low = mid + 1

        counts[ctx][bit] += 1
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


