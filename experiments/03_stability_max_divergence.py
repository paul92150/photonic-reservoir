import numpy as np
import matplotlib.pyplot as plt
from _setup_path import *
from src.reservoir.core import PhotonicReservoir


def estimate_divergence(reservoir, u1, u2, T=30):
    """
    Simulates the reservoir for two nearby inputs (batch=1 each) and returns
    the divergence curve: divergence(t) = log( distance(t) / distance(0) )
    """
    U_batch = np.stack([u1, u2], axis=0)  # shape (2, input_dim)
    states = reservoir.simulate_series(U_batch, T=T)  # shape (T, 2, N)

    s1 = states[:, 0, :]
    s2 = states[:, 1, :]
    distances = np.linalg.norm(s1 - s2, axis=1)
    distances = np.clip(distances, 1e-9, None)
    divergence = np.log(distances / distances[0])
    return divergence


def analyze_parameter_vary(param_to_vary='gamma'):
    N = 1024
    T = 30
    input_dim = 20
    fixed_params = {'I0': 1.5, 'gamma': 1.0, 'mu': 0.9}

    if param_to_vary == 'gamma':
        values = np.linspace(0.1, 1.0, 20)
    elif param_to_vary == 'I0':
        values = np.linspace(0.5, 1.5, 20)
    elif param_to_vary == 'mu':
        values = np.linspace(0.1, 0.9, 20)
    else:
        raise ValueError("param_to_vary must be one of 'gamma', 'I0', or 'mu'.")

    U_const = np.ones(input_dim)
    U_perturbed = U_const + np.random.normal(0, 0.001, size=U_const.shape)

    max_divergences = []
    param_values = []

    for val in values:
        params = fixed_params.copy()
        params[param_to_vary] = val

        reservoir = PhotonicReservoir(
            N=N,
            input_dim=input_dim,
            I0=params['I0'],
            gamma=params['gamma'],
            mu=params['mu'],
            n_bits_inner=8,
            n_bits_outer=10,
            inner_clip_min=-1.0,
            inner_clip_max=1.0,
            outer_clip_min=0.0,
            outer_clip_max=1.0,
            random_state=42, spectral_norm=True
        )

        divergence = estimate_divergence(reservoir, U_const, U_perturbed, T)
        max_div = np.max(divergence)
        max_divergences.append(max_div)
        param_values.append(val)
        print(f"{param_to_vary} = {val:.2f} → max divergence = {max_div:.4f}")

    return np.array(param_values), np.array(max_divergences)


def analyze_all_parameters():
    gamma_param, gamma_max_div = analyze_parameter_vary('gamma')
    I0_param, I0_max_div = analyze_parameter_vary('I0')
    mu_param, mu_max_div = analyze_parameter_vary('mu')

    fig, axs = plt.subplots(1, 3, figsize=(12, 5))

    axs[0].plot(gamma_param, gamma_max_div, marker='o', color='blue')
    axs[0].set_title("Impact of γ")
    axs[0].set_xlabel("γ")
    axs[0].set_ylabel("Max divergence (log)")
    axs[0].grid(True)

    axs[1].plot(I0_param, I0_max_div, marker='o', color='red')
    axs[1].set_title("Impact of I₀")
    axs[1].set_xlabel("I₀")
    axs[1].set_ylabel("Max divergence (log)")
    axs[1].grid(True)

    axs[2].plot(mu_param, mu_max_div, marker='o', color='purple')
    axs[2].set_title("Impact of μ")
    axs[2].set_xlabel("μ")
    axs[2].set_ylabel("Max divergence (log)")
    axs[2].grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    analyze_all_parameters()
