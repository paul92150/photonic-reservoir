import numpy as np
import matplotlib.pyplot as plt
from _setup_path import *
from src.reservoir.core import PhotonicReservoir 

def main():
    # Create a small reservoir for visualization
    reservoir = PhotonicReservoir(
        N=20, input_dim=3,
        I0=1.0, gamma=0.5, mu=0.1,
        n_bits_inner=8, n_bits_outer=10,
        inner_clip_min=-1.0, inner_clip_max=1.0,
        outer_clip_min=0.0, outer_clip_max=1.0,
        random_state=42
    )

    T = 30
    U = np.random.uniform(-1, 1, size=(T, reservoir.input_dim))

    states = []
    for t in range(T):
        x_t = reservoir.step(U[t])
        states.append(x_t.copy())
    states = np.array(states)

    # Plot a few neurons
    plt.figure(figsize=(10, 6))
    for neuron in [0, 1, 2, 3, 4]:
        plt.plot(states[:, neuron], label=f"Neuron {neuron}")
    plt.title("Photonic Reservoir Dynamics (Random Input)")
    plt.xlabel("Time Step")
    plt.ylabel("Activation")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
