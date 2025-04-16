import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
from _setup_path import *
from src.reservoir.core import PhotonicReservoir
from src.reservoir.utils import softmax, compute_Wout
from src.reservoir.hog_features import compute_hog_batch

# --- Classification Experiment for Varying Training Sizes ---
def classification_experiment_train_size(num_train, num_test=10000, use_hog=True):
    # Load MNIST data
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    from sklearn.utils import shuffle
    x_train, y_train = shuffle(x_train, y_train, random_state=52)
    
    # Select subsets
    x_train = x_train[:num_train]
    y_train = y_train[:num_train]
    x_test = x_test[:num_test]
    y_test = y_test[:num_test]

    
    # Preprocess: use HOG features if specified.
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
    
    print(f"Training size: {num_train}, Input feature dimension: {input_dim}")
    
    # Create the reservoir with fixed hyperparameters.
    reservoir = PhotonicReservoir(N=100, input_dim=input_dim, 
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
    
    # Compute output weights via ridge regression.
    W_out = compute_Wout(train_reservoir, Y_train_onehot, lmbda=1.5176331766409722e-05)
    
    # Predict on test set.
    Y_hat_test = softmax(test_reservoir @ W_out.T)
    y_pred_test = np.argmax(Y_hat_test, axis=1)
    acc_test = accuracy_score(y_test, y_pred_test)
    
    # Predict on training set.
    Y_hat_train = softmax(train_reservoir @ W_out.T)
    y_pred_train = np.argmax(Y_hat_train, axis=1)
    acc_train = accuracy_score(y_train, y_pred_train)
    
    print(f"Précision sur MNIST (HOG) :")
    print(f"  → Entraînement : {acc_train * 100:.2f}%")
    print(f"  → Test        : {acc_test * 100:.2f}%")
    
    return acc_train, acc_test

# --- Main Loop: Run over various training set sizes and plot results ---
if __name__ == "__main__":
    # Define a list of training set sizes.
    train_sizes = [100, 200, 500, 800, 1000, 2000, 5000, 8000, 60000]
    train_accs = []
    test_accs = []
    
    for num_train in train_sizes:
        print("\n-------------------------------------------")
        print(f"Training size = {num_train}")
        acc_train, acc_test = classification_experiment_train_size(num_train, num_test=int(num_train/5), use_hog=True)
        print(f'Test accuracy = {acc_test * 100:.2f}%')
        train_accs.append(acc_train)
        test_accs.append(acc_test)
    
    # Plot training and test accuracy vs. training set size.
    plt.figure(figsize=(8,6))
    plt.plot(train_sizes, [a * 100 for a in train_accs], 'o-', label='Training Accuracy (%)')
    plt.plot(train_sizes, [a * 100 for a in test_accs], 's-', label='Test Accuracy (%)')
    plt.xscale('log')
    plt.xlabel('Training Set Size (log scale)')
    plt.ylabel('Accuracy (%)')
    plt.title('Effect of Training Set Size on Accuracy (HOG)')
    plt.legend()
    plt.grid(True, which="both", ls="--", lw=0.5)
    plt.show()