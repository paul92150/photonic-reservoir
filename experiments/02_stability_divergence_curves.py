import numpy as np
import matplotlib.pyplot as plt
from _setup_path import *
from src.reservoir.core import PhotonicReservoir

def simulate_series(reservoir, u, T=30):
    """
    Run the reservoir for T steps with fixed input u.
    Returns states of shape (T, N).
    """
    states = []
    reservoir.reset_state()
    for _ in range(T):
        x = reservoir.step(u)
        states.append(x.copy())
    return np.array(states)

def estimate_divergence(reservoir, u1, u2, T=30):
    """
    Simule deux entrées proches et retourne la divergence log(||x1 - x2||).
    """
    reservoir.reset_state()
    s1 = simulate_series(reservoir, u1, T)

    reservoir.reset_state()
    s2 = simulate_series(reservoir, u2, T)

    distances = np.linalg.norm(s1 - s2, axis=1)
    distances = np.clip(distances, 1e-9, None)
    divergence = np.log(distances / distances[0])
    return divergence


def analyze_all_parameters_evolution():
    param_sets = {
        "gamma": np.linspace(0.1, 1.0, 10),
        "I0": np.linspace(0.5, 1.5, 10),
        "mu": np.linspace(0.1, 0.9, 10)
    }

    fixed_params = {
        "gamma": 0.5,
        "I0": 1.0,
        "mu": 0.3
    }

    input_dim = 20
    N = 1024
    T = 30
    U_const = np.ones(input_dim)
    U_perturbed = U_const + np.random.normal(0, 0.001, size=U_const.shape)

    fig, axs = plt.subplots(1, 3, figsize=(16, 5), sharey=True)

    for idx, (param_name, values) in enumerate(param_sets.items()):
        for val in values:
            params = fixed_params.copy()
            params[param_name] = val

            reservoir = PhotonicReservoir(
                N=N,
                input_dim=input_dim,
                gamma=params["gamma"],
                I0=params["I0"],
                mu=params["mu"],
                random_state=42, 
            )

            divergence = estimate_divergence(reservoir, U_const, U_perturbed, T)
            axs[idx].plot(range(1, T+1), divergence, label=f"{param_name}={val:.2f}", alpha=0.7)

        axs[idx].set_title(f"Impact of {param_name}")
        axs[idx].set_xlabel("Time Step")
        axs[idx].grid(True)
        if idx == 0:
            axs[idx].set_ylabel("log(Distance / Distance₀)")
        axs[idx].legend(fontsize=8)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    analyze_all_parameters_evolution()
