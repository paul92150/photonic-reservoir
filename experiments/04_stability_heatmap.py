import numpy as np
import matplotlib.pyplot as plt
import optuna
import pandas as pd
from scipy.interpolate import griddata
from _setup_path import *
from src.reservoir.core import PhotonicReservoir

def estimate_divergence(reservoir, u1, u2, T=30):
    U_batch = np.stack([u1, u2], axis=0)
    states = reservoir.simulate_series(U_batch, T=T)
    s1, s2 = states[:, 0, :], states[:, 1, :]
    distances = np.linalg.norm(s1 - s2, axis=1)
    distances = np.clip(distances, 1e-9, None)
    divergence = np.log(distances / distances[0])
    return np.max(divergence)

def objective(trial):
    gamma_val = trial.suggest_float("gamma", 0.1, 1.0)
    I0_val = trial.suggest_float("I0", 0.5, 1.5)
    mu_val = trial.suggest_float("mu", 0.1, 0.9)

    N, T, input_dim = 1024, 30, 20
    U_const = np.ones(input_dim)
    U_perturbed = U_const + np.random.normal(0, 0.001, size=input_dim)

    reservoir = PhotonicReservoir(
        N=N, input_dim=input_dim, I0=I0_val, gamma=gamma_val, mu=mu_val,
        n_bits_inner=8, n_bits_outer=10,
        inner_clip_min=-1.0, inner_clip_max=1.0,
        outer_clip_min=0.0, outer_clip_max=1.0,
        nonlin='sin', random_state=42
    )

    max_div = estimate_divergence(reservoir, U_const, U_perturbed, T)
    print(f"Trial: γ={gamma_val:.3f}, I₀={I0_val:.3f}, μ={mu_val:.3f} → max div = {max_div:.4f}")
    return max_div

def visualize_results(study):
    df = study.trials_dataframe()
    fixed_mu = study.best_params["mu"]
    mask = np.isclose(df["params_mu"].astype(float), fixed_mu, atol=0.05)
    
    gamma_vals = df["params_gamma"].astype(float)[mask]
    I0_vals = df["params_I0"].astype(float)[mask]
    max_div_vals = df["value"].astype(float)[mask]

    grid_gamma = np.linspace(0.1, 1.0, 50)
    grid_I0 = np.linspace(0.5, 1.5, 50)
    Gamma, I0_grid = np.meshgrid(grid_gamma, grid_I0)
    grid_max_div = griddata((gamma_vals, I0_vals), max_div_vals, (Gamma, I0_grid), method='cubic')

    plt.figure(figsize=(10, 8))
    plt.contourf(Gamma, I0_grid, grid_max_div, 20, cmap='viridis')
    plt.colorbar(label="Max divergence (log)")
    plt.xlabel("γ")
    plt.ylabel("I₀")
    plt.title(f"Heatmap: max divergence vs γ and I₀ (μ ≈ {fixed_mu:.2f})")
    plt.tight_layout()
    plt.show()

def main():
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=500)

    print("Best parameters:")
    print(study.best_params)
    print(f"Best divergence: {study.best_value:.4f}")
    
    visualize_results(study)

if __name__ == "__main__":
    main()
