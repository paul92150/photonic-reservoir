# src/reservoir/hog_features.py

import numpy as np
from skimage.feature import hog

def compute_hog_batch(X, 
                      pixels_per_cell=(5, 5), 
                      cells_per_block=(5, 5), 
                      orientations=6,
                      verbose=False):
    """
    Compute HOG (Histogram of Oriented Gradients) features for a batch of images.

    Parameters
    ----------
    X : np.ndarray
        Array of shape (n_samples, height, width) containing grayscale images.
    pixels_per_cell : tuple of int
        Size (in pixels) of a cell.
    cells_per_block : tuple of int
        Number of cells in each block.
    orientations : int
        Number of orientation bins.
    verbose : bool
        Whether to print progress messages during computation.

    Returns
    -------
    hog_features : np.ndarray
        Array of shape (n_samples, n_features) containing HOG descriptors.
    """
    hog_features = []
    for i, img in enumerate(X):
        feat = hog(img,
                   orientations=orientations,
                   pixels_per_cell=pixels_per_cell,
                   cells_per_block=cells_per_block,
                   block_norm='L2-Hys',
                   feature_vector=True)
        hog_features.append(feat)

        if verbose and i % 1000 == 0:
            print(f"HOG computed for {i}/{len(X)} images")

    return np.array(hog_features)
