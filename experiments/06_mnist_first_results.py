import numpy as np
from tensorflow.keras.datasets import mnist
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
from _setup_path import *
from src.reservoir.core import PhotonicReservoir
from src.reservoir.utils import softmax, compute_Wout
from src.reservoir.hog_features import compute_hog_batch

# --- MNIST Classification ---
def classification_experiment(use_hog=False):

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Subsample if needed
    num_train, num_test = 60000, 10000
    x_train, y_train = x_train[:num_train], y_train[:num_train]
    x_test, y_test = x_test[:num_test], y_test[:num_test]

    if use_hog:

        X_train_features = compute_hog_batch(x_train)
        X_test_features = compute_hog_batch(x_test)
        input_dim = X_train_features.shape[1]
    else:
        X_train_features = x_train.reshape((-1, 28 * 28), order='F').astype(np.float32)
        X_test_features = x_test.reshape((-1, 28 * 28), order='F').astype(np.float32)
        X_train_features = 2 * (X_train_features / 255.0) - 1.0
        X_test_features = 2 * (X_test_features / 255.0) - 1.0
        input_dim = 784

    print(f"Input dimension: {input_dim} | Dataset: {'HOG' if use_hog else 'Raw'}")

    # Reservoir
    reservoir = PhotonicReservoir(
        N=1024,
        input_dim=input_dim,
        I0=1.133,
        gamma=0.983,
        mu=0.765,
        nonlin='sin',
        spectral_norm=True,
        n_bits_inner=8,
        n_bits_outer=10,
        inner_clip_min=-1.0,
        inner_clip_max=1.0,
        outer_clip_min=0.0,
        outer_clip_max=1.0,
        random_state=42
    )

    X_train_res = reservoir.transform(X_train_features)
    X_test_res = reservoir.transform(X_test_features)

    encoder = OneHotEncoder(sparse_output=False)
    Y_train_onehot = encoder.fit_transform(y_train.reshape(-1, 1))

    W_out = compute_Wout(X_train_res, Y_train_onehot, lmbda=1e-6)

    Y_pred_probs = softmax(X_test_res @ W_out.T)
    y_pred = np.argmax(Y_pred_probs, axis=1)

    acc = accuracy_score(y_test, y_pred)
    print(f"Test accuracy ({'HOG' if use_hog else 'Raw'}): {acc * 100:.2f}%")
    return acc


if __name__ == "__main__":
    print("MNIST Classification - Raw Input")
    acc_raw = classification_experiment(use_hog=False)

    print("\nMNIST Classification - HOG Input")
    acc_hog = classification_experiment(use_hog=True)
