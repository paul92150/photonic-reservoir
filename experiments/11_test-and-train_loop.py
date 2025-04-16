import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
from _setup_path import *
from src.reservoir.core import PhotonicReservoir
from src.reservoir.utils import softmax, compute_Wout
from src.reservoir.hog_features import compute_hog_batch

# --- Run Experiment for a Given Reservoir Size and Training Set Size ---
def run_experiment(N, num_train, num_test=10000, use_hog=True):
    # Load and shuffle MNIST data.
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, y_train = shuffle(x_train, y_train, random_state=42)
    
    # Subset the data.
    x_train = x_train[:num_train]
    y_train = y_train[:num_train]
    x_test = x_test[:num_test]
    y_test = y_test[:num_test]
    
    # Extract features.
    if use_hog:
        X_train_features = compute_hog_batch(x_train)
        X_test_features = compute_hog_batch(x_test)
        input_dim = X_train_features.shape[1]
    else:
        X_train_features = np.array([img.flatten(order='F') for img in x_train]).astype(np.float32)
        X_test_features = np.array([img.flatten(order='F') for img in x_test]).astype(np.float32)
        X_train_features = 2 * (X_train_features / 255.0) - 1.0
        X_test_features = 2 * (X_test_features / 255.0) - 1.0
        input_dim = X_train_features.shape[1]
    
    # Create a new reservoir for each experiment.
    reservoir = PhotonicReservoir(N=N, input_dim=input_dim, 
                                  I0=0.9923608833582865, 
                                  gamma=0.11054370609219216, 
                                  mu=0.36042090297660445,
                                  n_bits_inner=8, n_bits_outer=10,
                                  inner_clip_min=-1.0, inner_clip_max=1.0,
                                  outer_clip_min=0.0, outer_clip_max=1.0,
                                  nonlin='sin', random_state=42)
    
    # Compute reservoir features.
    train_reservoir = reservoir.transform(X_train_features)
    test_reservoir = reservoir.transform(X_test_features)
    
    # One-hot encode labels.
    encoder = OneHotEncoder()
    Y_train_onehot = encoder.fit_transform(y_train.reshape(-1, 1))
    
    # Compute readout weights.
    W_out = compute_Wout(train_reservoir, Y_train_onehot, lmbda=1.5176331766409722e-07)
    
    # Predictions and accuracies.
    Y_hat_train = softmax(train_reservoir @ W_out.T)
    y_pred_train = np.argmax(Y_hat_train, axis=1)
    acc_train = accuracy_score(y_train, y_pred_train)
    
    Y_hat_test = softmax(test_reservoir @ W_out.T)
    y_pred_test = np.argmax(Y_hat_test, axis=1)
    acc_test = accuracy_score(y_test, y_pred_test)
    
    norm_W_out = np.linalg.norm(W_out)
    
    return acc_train, acc_test, norm_W_out

# --- Main Function: Loop Over Reservoir Sizes and Training Set Sizes, Then Plot ---
if __name__ == "__main__":
    # Define reservoir sizes and training set sizes.
    reservoir_sizes = [200, 500, 800, 1000, 1500, 2000]
    train_sizes = [100, 200, 500, 1000, 2000, 3000 ]
    
    # Create dictionaries to store results for each reservoir size.
    results_acc_test = {N: [] for N in reservoir_sizes}
    results_norm = {N: [] for N in reservoir_sizes}
    
    # Loop over reservoir sizes.
    for N in reservoir_sizes:
        print(f"\n=== Reservoir size: {N} ===")
        # For each reservoir size, loop over training set sizes.
        for num_train in train_sizes:
            print(f"Training set size: {num_train}")
            _, acc_test, norm_W = run_experiment(N, num_train, num_test=10000, use_hog=True)
            results_acc_test[N].append(acc_test * 100)  # Convert to percentage
            results_norm[N].append(norm_W)
    
    # Plot Test Accuracy vs Training Set Size.
    plt.figure(figsize=(10, 5))
    for N in reservoir_sizes:
        plt.plot(train_sizes, results_acc_test[N], marker='o', label=f'N={N}')
    plt.xscale('log')
    plt.xlabel('Training Set Size (log scale)')
    plt.ylabel('Test Accuracy (%)')
    plt.title('Test Accuracy vs Training Set Size for Different Reservoir Sizes')
    plt.legend()
    plt.grid(True, which="both", ls="--", lw=0.5)
    plt.show()
    
    # Plot Norm of W_out vs Training Set Size.
    plt.figure(figsize=(10, 5))
    for N in reservoir_sizes:
        plt.plot(train_sizes, results_norm[N], marker='s', label=f'N={N}')
    plt.xscale('log')
    plt.xlabel('Training Set Size (log scale)')
    plt.ylabel('||W_out|| (L2 Norm)')
    plt.title('Norm of W_out vs Training Set Size for Different Reservoir Sizes')
    plt.legend()
    plt.grid(True, which="both", ls="--", lw=0.5)
    plt.show()
