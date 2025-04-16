# src/reservoir/utils.py

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder

# --- Quantization function ---
def quantize(x, n_bits, clip_min, clip_max):
    levels = 2 ** n_bits
    x_clipped = np.clip(x, clip_min, clip_max)
    x_scaled = (x_clipped - clip_min) / (clip_max - clip_min) * (levels - 1)
    x_quant = np.round(x_scaled)
    x_reconstructed = x_quant / (levels - 1) * (clip_max - clip_min) + clip_min
    return x_reconstructed

def compute_Wout(X, Y, lmbda=1e-6):
    """
    Ridge regression (Tikhonov regularization) to solve for W_out.
    
    Parameters:
    - X : ndarray, shape (n_samples, n_features)
        Reservoir states
    - Y : ndarray, shape (n_samples, n_classes)
        One-hot encoded labels
    - lmbda : float
        Regularization strength
    
    Returns:
    - W_out : ndarray, shape (n_classes, n_features)
        Optimal readout weights
    """
    XT_X = X.T @ X
    regularized = XT_X + lmbda * np.eye(X.shape[1])
    W_out = np.linalg.solve(regularized, X.T @ Y)
    return W_out.T


def softmax(z):
    """
    Apply softmax to a batch of logits.
    
    Parameters:
    - z : ndarray, shape (batch_size, num_classes)
    
    Returns:
    - probabilities : ndarray, shape (batch_size, num_classes)
    """
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def compute_accuracy(y_true, logits):
    """
    Compute accuracy from logits (before softmax).
    
    Parameters:
    - y_true : ndarray, shape (n_samples,)
        True integer class labels.
    - logits : ndarray, shape (n_samples, n_classes)
        Predicted logits (pre-softmax).
    
    Returns:
    - accuracy : float
    """
    y_pred = np.argmax(logits, axis=1)
    return accuracy_score(y_true, y_pred)


def one_hot_encode(y):
    """
    One-hot encode class labels.
    
    Parameters:
    - y : ndarray, shape (n_samples,)
        Integer class labels.
    
    Returns:
    - onehot : ndarray, shape (n_samples, n_classes)
    """
    encoder = OneHotEncoder(sparse_output=False)
    return encoder.fit_transform(y.reshape(-1, 1))


def compute_condition_number(X):
    """
    Compute the condition number of XᵀX.
    
    Parameters:
    - X : ndarray, shape (n_samples, n_features)
    
    Returns:
    - cond_number : float
    """
    XT_X = X.T @ X
    return np.linalg.cond(XT_X)
