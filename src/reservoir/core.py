import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from src.reservoir.utils import quantize

class PhotonicReservoir:
    """
    A fast and flexible photonic-inspired reservoir computing model.
    """

    def __init__(self, N=1024, input_dim=784, I0=1.0, gamma=0.5, mu=0.1,
                 n_bits_inner=8, n_bits_outer=10,
                 inner_clip_min=-1.0, inner_clip_max=1.0,
                 outer_clip_min=0.0, outer_clip_max=1.0,
                 nonlin='sin', random_state=42, spectral_norm=False):
        self.N = N
        self.input_dim = input_dim
        self.I0 = I0
        self.gamma = gamma
        self.mu = mu
        self.n_bits_inner = n_bits_inner
        self.n_bits_outer = n_bits_outer
        self.inner_clip_min = inner_clip_min
        self.inner_clip_max = inner_clip_max
        self.outer_clip_min = outer_clip_min
        self.outer_clip_max = outer_clip_max
        self.nonlin = nonlin
        self.rng = np.random.default_rng(random_state)

        W_res_full = self.rng.normal(0, 1, (N, N))
        mask = self.rng.random((N, N)) < mu
        self.W_res = W_res_full * mask

        if spectral_norm:
            eigvals = np.linalg.eigvals(self.W_res)
            spectral_radius = np.max(np.abs(eigvals))
            if spectral_radius != 0:
                self.W_res /= spectral_radius

        self.W_in = gamma * self.rng.normal(0, 1, (N, input_dim))
        self.reset_state()

    def _nonlinearity(self, x):
        if self.nonlin == 'sin':
            return np.sin(x)**2
        elif self.nonlin == 'tanh':
            return np.tanh(x)**2
        else:
            raise ValueError(f"Unsupported nonlinearity: {self.nonlin}")

    def step(self, u):
        activation = self.W_res @ self.x + self.W_in @ u
        activation_q = quantize(activation, self.n_bits_inner,
                                self.inner_clip_min, self.inner_clip_max)
        nonlinear = self._nonlinearity(activation_q)
        new_state_raw = self.I0 * nonlinear
        new_state = quantize(new_state_raw, self.n_bits_outer,
                             self.outer_clip_min, self.outer_clip_max)
        self.x = new_state
        return self.x

    def transform(self, U):
        features = []
        for u in U:
            self.reset_state()
            features.append(self.step(u).copy())
        return np.array(features)

    def simulate_series(self, U, T):
        batch = U.shape[0]
        X_state = np.full((batch, self.N), 0.1)
        states = []
        for _ in range(T):
            activation = np.einsum('ij,bj->bi', self.W_res, X_state) + np.dot(U, self.W_in.T)
            activation_q = quantize(activation, self.n_bits_inner, self.inner_clip_min, self.inner_clip_max)
            nonlinear = self._nonlinearity(activation_q)
            new_state_raw = self.I0 * nonlinear
            quant_state = quantize(new_state_raw, self.n_bits_outer, self.outer_clip_min, self.outer_clip_max)
            X_state = quant_state
            states.append(X_state.copy())
        return np.array(states)

    def reset_state(self):
        self.x = np.full(self.N, 0.1)