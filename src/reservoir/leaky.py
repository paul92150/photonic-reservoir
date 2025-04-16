import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
from src.reservoir.utils import quantize 



class PhotonicLeakyReservoir:
    """
    Photonic reservoir with leaky integration:
    x(t+1) = (1 - α)x(t) + α * quant_{10}[ I₀ * sin²( quant_{8}[W_res * x(t) + W_in * u(t)] ) ]
    """

    def __init__(self, N=1024, input_dim=10, I0=0.3, gamma=0.7, mu=0.3,
                 alpha=0.5, T=50, nonlin='sin',
                 n_bits_inner=8, n_bits_outer=10,
                 inner_clip_min=-1.0, inner_clip_max=1.0,
                 outer_clip_min=0.0, outer_clip_max=1.0,
                 random_state=42):
        self.N = N
        self.input_dim = input_dim
        self.I0 = I0
        self.gamma = gamma
        self.mu = mu
        self.alpha = alpha
        self.T = T
        self.nonlin = nonlin
        self.n_bits_inner = n_bits_inner
        self.n_bits_outer = n_bits_outer
        self.inner_clip_min = inner_clip_min
        self.inner_clip_max = inner_clip_max
        self.outer_clip_min = outer_clip_min
        self.outer_clip_max = outer_clip_max
        self.rng = np.random.default_rng(random_state)

        # Initialize W_res with sparsity
        W_res_full = self.rng.normal(loc=0.0, scale=1.0, size=(N, N))
        mask = self.rng.random((N, N)) < mu
        self.W_res = W_res_full * mask

        # Initialize W_in
        self.W_in = gamma * self.rng.normal(loc=0.0, scale=1.0, size=(N, input_dim))

        # Initialize state
        self.x = np.full(N, 0.1)

    def step(self, u):
        """Single input step update with leaky integration."""
        activation = self.W_res @ self.x + self.W_in @ u
        activation_q = quantize(activation, self.n_bits_inner, self.inner_clip_min, self.inner_clip_max)
        nonlinear = np.sin(activation_q)**2 if self.nonlin == 'sin' else np.tanh(activation_q)**2
        raw = self.I0 * nonlinear
        quant = quantize(raw, self.n_bits_outer, self.outer_clip_min, self.outer_clip_max)
        self.x = (1 - self.alpha) * self.x + self.alpha * quant
        return self.x

    def reset_state(self):
        """Reset reservoir state."""
        self.x = np.full(self.N, 0.1)

    def simulate_series(self, U):
        """
        Simulate reservoir over T steps for batch input U (shape: (batch, input_dim)).
        Returns array of shape (T, batch, N).
        """
        batch = U.shape[0]
        X_state = np.full((batch, self.N), 0.1)
        states = []
        for _ in range(self.T):
            activation = np.einsum('ij,bj->bi', self.W_res, X_state) + U @ self.W_in.T
            activation_q = quantize(activation, self.n_bits_inner, self.inner_clip_min, self.inner_clip_max)
            nonlinear = np.sin(activation_q)**2 if self.nonlin == 'sin' else np.tanh(activation_q)**2
            raw = self.I0 * nonlinear
            quant = quantize(raw, self.n_bits_outer, self.outer_clip_min, self.outer_clip_max)
            X_state = (1 - self.alpha) * X_state + self.alpha * quant
            states.append(X_state.copy())
        return np.array(states)


def convergence_time(time_series, threshold=1e-3, consecutive=3):
    """
    Compute convergence time of a neuron.
    Returns first time t where abs differences < threshold for `consecutive` steps.
    """
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
