import numpy as np
import matplotlib.pyplot as plt
from _setup_path import *
from src.reservoir.leaky import PhotonicLeakyReservoir 

def convergence_time(time_series, threshold=1e-3, consecutive=3):
    diffs = np.abs(np.diff(time_series))
    count = 0
    for t, d in enumerate(diffs):
        if d < threshold:
            count += 1
            if count >= consecutive:
                return t - consecutive + 2
        else:
            count = 0
    return len(time_series)

def analyze_alpha_convergence():
    N = 1024
    T = 50
    input_dim = 10
    I0 = 0.3
    gamma = 0.7
    mu = 0.3
    alpha_values = np.linspace(0.1, 1.0, 40)

    U = np.ones((1, input_dim))
    np.random.seed(42)
    selected_neurons = np.random.choice(N, 10, replace=False)
    avg_conv_times = []

    n_alpha = len(alpha_values)
    rows = int(np.ceil(n_alpha / 2))
    fig, axs = plt.subplots(rows, 2, figsize=(14, 2 * rows), sharex=True)
    axs = axs.flatten()

    for i, alpha_val in enumerate(alpha_values):
        reservoir = PhotonicLeakyReservoir(
            N=N, input_dim=input_dim, I0=I0, gamma=gamma, mu=mu,
            n_bits_inner=8, n_bits_outer=10,
            inner_clip_min=-1.0, inner_clip_max=1.0,
            outer_clip_min=0.0, outer_clip_max=1.0,
            nonlin='sin', random_state=42,
            alpha=alpha_val  # Leaky integration parameter
        )

        # Simulate the reservoir for T time steps
        states = []
        reservoir.reset_state()
        for _ in range(T):
            x = reservoir.step(U[0])
            states.append(x.copy())
        states = np.array(states)  # Shape: (T, N)

        # Plot selected neurons
        ax = axs[i]
        conv_times = []
        time_steps = np.arange(1, T + 1)
        for neuron in selected_neurons:
            neuron_state = states[:, neuron]
            ax.plot(time_steps, neuron_state, label=f"Neuron {neuron}")
            conv_time = convergence_time(neuron_state)
            conv_times.append(conv_time)

        avg_conv = np.mean(conv_times)
        avg_conv_times.append(avg_conv)

        ax.set_title(f"α = {alpha_val:.2f} (avg t = {avg_conv:.1f})")
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Activation")
        ax.grid(True)
        if i == 0:
            ax.legend(fontsize=8)
        print(f"α = {alpha_val:.2f} → avg convergence time = {avg_conv:.2f} steps")

    # Hide unused subplots
    for j in range(i + 1, len(axs)):
        axs[j].axis('off')

    plt.tight_layout()
    plt.show()

    # Summary plot of average convergence time vs alpha
    plt.figure(figsize=(8, 6))
    plt.plot(alpha_values, avg_conv_times, marker='o')
    plt.xlabel("α")
    plt.ylabel("Average Convergence Time (steps)")
    plt.title("Effect of α on Neuron Convergence Time")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    analyze_alpha_convergence()