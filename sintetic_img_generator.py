import numpy as np
from scipy.ndimage import label

def generate_synthetic_map(n_classes, distribution='uniform', size=(512, 614), seed=None):
    """
    distribution: 'uniform', 'dominant', 'exponential', 'bimodal'
    """
    rng = np.random.default_rng(seed)
    
    # Define class probabilities
    if distribution == 'uniform':
        probs = np.ones(n_classes) / n_classes
        
    elif distribution == 'dominant':
        probs = np.ones(n_classes) * 0.3 / (n_classes - 1)
        probs[0] = 0.7
        
    elif distribution == 'exponential':
        probs = np.array([0.5 ** (i + 1) for i in range(n_classes)])
        probs[-1] += 1 - probs.sum()  # ajusta para que sume 1
        
    elif distribution == 'bimodal':
        probs = np.ones(n_classes) * 0.2 / (n_classes - 2)
        probs[0] = 0.4
        probs[1] = 0.4

    # Genera regiones espacialmente coherentes con Voronoi simple
    n_seeds = size[0] * size[1] // 500  # controla tamaño de regiones
    seed_points = rng.integers(0, [size[0], size[1]], size=(n_seeds, 2))
    seed_classes = rng.choice(n_classes, size=n_seeds, p=probs)
    
    rows, cols = np.mgrid[0:size[0], 0:size[1]]
    coords = np.stack([rows.ravel(), cols.ravel()], axis=1)
    
    from scipy.spatial import cKDTree
    tree = cKDTree(seed_points)
    _, idx = tree.query(coords)
    image = seed_classes[idx].reshape(size)
    
    return image

'''
distributions = ['uniform', 'dominant', 'exponential', 'bimodal']

import matplotlib.pyplot as plt
for dist in distributions:
    img = generate_synthetic_map(n_classes=5, distribution=dist, seed=42)



    plt.imshow(img, cmap='terrain')
    plt.title("Mapa de Clasificación Sintético")
    plt.colorbar(label="ID de Clase")
    plt.show()

'''