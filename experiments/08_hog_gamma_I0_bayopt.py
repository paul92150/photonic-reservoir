import numpy as np
import optuna
import optuna.visualization as vis
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.datasets import mnist
from _setup_path import *
from src.reservoir.core import PhotonicReservoir
from src.reservoir.utils import  softmax, compute_Wout
from src.reservoir.hog_features import compute_hog_batch


def objective(trial):
    # HOG hyperparameters
    pixels_per_cell = trial.suggest_int("pixels_per_cell", 2, 14)
    cells_per_block = trial.suggest_int("cells_per_block", 1, 14)
    orientations = trial.suggest_int("orientations", 3, 12)

    if pixels_per_cell * cells_per_block > 28:
        raise optuna.TrialPruned()

    # Reservoir hyperparameters
    gamma = trial.suggest_float("gamma", 0.1, 1.0)
    I0 = trial.suggest_float("I0", 0.5, 1.5)

    # Load MNIST
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, y_train = x_train[:10000], y_train[:10000]
    x_test, y_test = x_test[:2000], y_test[:2000]

    # Extract HOG features
    X_train = compute_hog_batch(x_train, (pixels_per_cell, pixels_per_cell), (cells_per_block, cells_per_block), orientations)
    X_test = compute_hog_batch(x_test, (pixels_per_cell, pixels_per_cell), (cells_per_block, cells_per_block), orientations)

    input_dim = X_train.shape[1]

    # Create reservoir
    reservoir = PhotonicReservoir(
        N=512, input_dim=input_dim, I0=I0, gamma=gamma, mu=0.765,
        n_bits_inner=8, n_bits_outer=10,
        inner_clip_min=-1.0, inner_clip_max=1.0,
        outer_clip_min=0.0, outer_clip_max=1.0,
        nonlin='sin', random_state=42
    )

    # Transform features
    X_train_res = reservoir.transform(X_train)
    X_test_res = reservoir.transform(X_test)

    # Train output weights
    encoder = OneHotEncoder(sparse_output=False)
    Y_train_onehot = encoder.fit_transform(y_train.reshape(-1, 1))
    W_out = compute_Wout(X_train_res, Y_train_onehot, lmbda=1e-6)

    # Evaluate accuracy
    y_pred = np.argmax(softmax(X_test_res @ W_out.T), axis=1)
    return accuracy_score(y_test, y_pred)


if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100)

    # Best results
    print("Best trial:")
    print(f"  Accuracy: {study.best_value * 100:.2f}%")
    print("  Hyperparameters:")
    for k, v in study.best_params.items():
        print(f"    {k}: {v}")

    # Visualizations
    vis_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(vis_dir, exist_ok=True)

    vis.plot_optimization_history(study).write_html(os.path.join(vis_dir, "hog_opt_history.html"))
    vis.plot_param_importances(study).write_html(os.path.join(vis_dir, "hog_param_importance.html"))
    vis.plot_slice(study).write_html(os.path.join(vis_dir, "hog_slice_plot.html"))
    vis.plot_parallel_coordinate(study).write_html(os.path.join(vis_dir, "hog_parallel.html"))
    vis.plot_contour(study).write_html(os.path.join(vis_dir, "hog_contour.html"))

    print(f"Visualizations saved in: {vis_dir}")
