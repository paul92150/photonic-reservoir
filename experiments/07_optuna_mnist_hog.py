import numpy as np
import optuna
from tensorflow.keras.datasets import mnist
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
from _setup_path import *
from src.reservoir.core import PhotonicReservoir
from src.reservoir.utils import softmax, compute_Wout
from src.reservoir.hog_features import compute_hog_batch
import optuna.visualization as vis

# --- Optuna Objective ---
def objective(trial):
    # Hyperparameter search space
    gamma = trial.suggest_float("gamma", 0.1, 1.0)
    I0 = trial.suggest_float("I0", 0.5, 1.5)
    mu = trial.suggest_float("mu", 0.1, 0.9)
    lmbda = trial.suggest_float("lambda", 1e-8, 1e-2, log=True)

    # Load data
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, y_train = x_train[:10000], y_train[:10000]
    x_test, y_test = x_test[:2000], y_test[:2000]

    # Preprocess using HOG
    X_train = compute_hog_batch(x_train)
    X_test = compute_hog_batch(x_test)
    input_dim = X_train.shape[1]

    # Reservoir creation
    reservoir = PhotonicReservoir(N=1024, input_dim=input_dim, I0=I0, gamma=gamma, mu=mu,
                                  inner_clip_min=-1.0, inner_clip_max=1.0,
                                  outer_clip_min=0.0, outer_clip_max=1.0,
                                  n_bits_inner=8, n_bits_outer=10,
                                  nonlin='sin', random_state=42, spectral_norm=True)

    # Transform features
    X_res_train = reservoir.transform(X_train)
    X_res_test = reservoir.transform(X_test)

    encoder = OneHotEncoder(sparse_output=False)
    Y_train_onehot = encoder.fit_transform(y_train.reshape(-1, 1))

    W_out = compute_Wout(X_res_train, Y_train_onehot, lmbda=lmbda)
    Y_pred = softmax(X_res_test @ W_out.T)
    y_pred = np.argmax(Y_pred, axis=1)

    return accuracy_score(y_test, y_pred)

# --- Run Optimization ---
if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=4)

    print("\nBest Trial:")
    print(f"  Accuracy: {study.best_value * 100:.2f}%")
    print("  Hyperparameters:")
    for k, v in study.best_params.items():
        print(f"    {k}: {v}")

    # Save plots
    os.makedirs("experiments/optuna_viz", exist_ok=True)
    vis.plot_optimization_history(study).write_html("experiments/optuna_viz/history.html")
    vis.plot_param_importances(study).write_html("experiments/optuna_viz/importances.html")
    vis.plot_slice(study).write_html("experiments/optuna_viz/slice.html")
    vis.plot_parallel_coordinate(study).write_html("experiments/optuna_viz/parallel.html")
    vis.plot_contour(study).write_html("experiments/optuna_viz/contour.html")
    print("Interactive plots saved to 'experiments/optuna_viz'")
