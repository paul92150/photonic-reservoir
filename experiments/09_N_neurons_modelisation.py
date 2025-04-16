import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# -----------------------------
# 1. Data
# -----------------------------
# Reservoir sizes (Nₓ) and measured accuracy (%)
N = np.array([1000, 2000, 4000, 8000])
accuracy = np.array([97.90, 98.49, 98.76, 98.86])

# -----------------------------
# 2. Define the generalized exponential model
# -----------------------------
def gen_exp_model(x, A_max, lam, p):
    """
    Generalized Exponential Model:
    
      A(x) = A_max * (1 - exp(- (x/lam)^p ))
    
    - A_max is the asymptotic maximum accuracy.
    - lam (lambda) controls the horizontal scaling.
    - p adjusts the steepness of the rising phase.
    """
    return A_max * (1 - np.exp(-(x / lam)**p))

# -----------------------------
# 3. Fit the model to the data
# -----------------------------
# Choose initial guesses and bounds.
# Here we expect A_max to be near 99 (%). For lam, note that if lam is too small,
# the function will saturate too quickly. p modulates the curve shape.
p0_exp = [99.0, 3, 0.5]  # initial guess: A_max ~99, lam ~3000, p ~0.5
bounds_exp = ([97.5, 2, 0.1], [100.0, 20000, 5.0])  # reasonable ranges

params_exp, _ = curve_fit(gen_exp_model, N, accuracy, p0=p0_exp, bounds=bounds_exp)

# Generate predictions over an extended range of N values (for example, from 1000 to 20000)
N_ext = np.linspace(1000, 20000, 500)
acc_exp = gen_exp_model(N_ext, *params_exp)

# -----------------------------
# 4. Plot the data and the fitted model
# -----------------------------
plt.figure(figsize=(10, 6))
plt.scatter(N, accuracy, color='black', label='Data', zorder=5)
plt.plot(N_ext, acc_exp, label='Generalized Exponential Fit', linestyle='--')
plt.xlabel('Reservoir Size (Nₓ)')
plt.ylabel('Accuracy (%)')
plt.title('Fit using A(N)=A_max*(1 - exp(-(N/λ)^p))')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# -----------------------------
# 5. Report the fitted parameters
# -----------------------------
print("Generalized Exponential Model Parameters:")
print("A_max = {:.3f}".format(params_exp[0]))
print("λ     = {:.1f}".format(params_exp[1]))
print("p     = {:.3f}".format(params_exp[2]))