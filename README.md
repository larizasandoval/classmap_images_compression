# Analyzing Lossless Compression of Classification Map Images

This repository contains the code, corpora, and experiments developed for the Master's
thesis *"Analyzing Lossless Compression of Classification Map Images"* (Universitat
Autònoma de Barcelona, MSc in Research and Innovation for Computer-based Science and
Engineering, 2026).

- **Author:** Lariza Sandoval Cruz


The thesis replicates and analyzes the classification-map compression pipeline proposed by
Xie and Klimesh [1] — piecewise pixel prediction combined with pattern-based context
modeling and interleaved entropy coding — and evaluates three modifications aimed at
improving its performance:

1. **Two-pass (offline) probability model** for the entropy coder, as an alternative to the
   original online adaptive model.
2. **Two-question predictor**, a simplified variant of the piecewise predictor limited to
   the first two neighboring queries.
3. **Entropy coding of raw values** (pixels not matched by the predictor), quantifying the
   theoretical bit-rate gain achievable over their current fixed-length representation.

All three proposals are evaluated on two corpora: the five classification maps originally
used by Xie and Klimesh [1], and a new validation corpus of 70 maps built for this work from
five additional sensors and a different classifier (see [Corpora](#corpora) below).


## Corpora

| Corpus | Sensor | No. of images | Classifier |
|---|---|---|---|
| Original [1] | AVIRIS | 3 | SVM / spectral clustering |
| Validation (this work) | CASI, Hyperion, IASI L1C, Landsat, AIRS, AVIRIS | 14 (70 maps total) | k-means |

For each image in both corpora, classification maps were generated at five class levels
(n = 4, 7, 9, 17, 32). Full details on corpus construction are given in the thesis
(Section IV.1).

## Pipeline overview

The pipeline follows three sequential stages, described in full in the thesis (Section II):

1. **Piecewise pixel prediction** — for each pixel, up to four binary decisions determine
   whether it matches one of its four immediate neighbors (left, upper, upper-left,
   upper-right); pixels matching none are transmitted as raw values.
2. **Pattern-based context modeling** — each decision bit is assigned to one of 37 contexts
   based on the geometric pattern formed by the neighboring pixels.
3. **Entropy coding** — decision bits are compressed using an adaptive arithmetic encoder.
