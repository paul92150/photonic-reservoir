# 🔬 Photonic-Inspired Reservoir Computing

A neuromorphic machine learning system simulating a **quantized photonic reservoir**, designed for stability analysis and image classification tasks. Includes custom experiments on convergence, divergence, classification accuracy on MNIST (with and without HOG), and Bayesian optimization using Optuna.

![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)

---

## 📁 Project Structure

```
photonic-reservoir/
├── src/
│   └── reservoir/
│       ├── core.py              # Core reservoir model
│       ├── leaky.py             # Leaky reservoir model (alpha control)
│       ├── utils.py             # Utility functions (metrics, plotting)
│       └── hog_features.py      # HOG preprocessing utilities
├── experiments/                 # All experiment scripts (numbered)
│   ├── 01_random_dynamics_demo.py
│   ├── 02_stability_divergence_curves.py
│   ├── ...
├── report/
│   └── report.pdf               # Final 26-page academic report
├── figures/                     #  Key figures and visualizations
├── requirements.txt
├── LICENSE
└── .gitignore
```

---

## 💡 Overview

This project was developed as part of a neuromorphic computing course at **CentraleSupélec** (ST7). It implements a **fast, quantized simulation of a photonic-inspired reservoir** using NumPy, and explores:

- Stability & divergence using pseudo-Lyapunov curves  
- Convergence speed under leaky integration  
- MNIST classification using raw and HOG features  
- Hyperparameter optimization (γ, I₀, μ) via Bayesian methods  
- End-to-end benchmarking with a ridge regression readout  

---

## ⚙️ Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/Paul92150/photonic-reservoir.git
cd photonic-reservoir
pip install -r requirements.txt
```

---

## 🚀 Running Experiments

All experiments are located in the `experiments/` folder.  
To run any of them, use the following command from the root of the project:

```bash
python experiments/05_leaky_convergence_analysis.py
```

### 📂 Highlights

| File                                | Description                                      |
|-------------------------------------|--------------------------------------------------|
| `02_stability_divergence_curves.py` | Pseudo-Lyapunov divergence over time            |
| `04_stability_heatmap.py`           | Heatmap of max divergence vs (γ, I₀)            |
| `05_leaky_convergence_analysis.py`  | Convergence time analysis with leaky integration |
| `06_mnist_first_results.py`         | MNIST classification with raw / HOG features    |
| `07_optuna_mnist_hog.py`            | Bayesian optimization of reservoir parameters   |
| `08_hog_gamma_I0_bayopt.py`         | Joint HOG + reservoir hyperparameter search     |

---

## 📊 Results

- ✅ 98.86% test accuracy on MNIST (HOG + tuned reservoir with 8000 neurons)  
- 📉 Leaky integration improves convergence times (α ≈ 0.4–0.7 optimal)  
- 📈 Clear stability-chaos transitions observed in divergence plots  

➡️ See all figures and discussion in `report/report.pdf`.

---

## 🔧 Key Features

- Fully quantized model (inner & outer bit-depth, clippings)
- Reservoir size customizable from 256 to 8000+
- Supports one-shot (`transform`) and temporal (`simulate_series`) dynamics
- Compatible with Optuna for full pipeline optimization
- Lightweight and fast: built with NumPy, scikit-learn, scikit-image

---

## 📄 License

This project is licensed under the MIT License.  
Feel free to use, modify, and build upon it. See [LICENSE](LICENSE).

---

## 🧠 Summary (Non-Technical)

This project simulates how a brain-like wave-based system processes images.  
It mimics **photonic reservoirs** (light-based neural networks) with a digital simulation.  
The goal is to explore their behavior and test their ability to recognize handwritten digits (MNIST).  
Despite its simplicity, the model reaches ~99% accuracy and exhibits fascinating properties like **memory**, **chaos**, and **stability**.

---

## 👤 Author

**Paul Lemaire**  
🎓 CentraleSupélec Student (Gap Year – AI/ML)  
📫 Contact: paul.lemaire@student-cs.fr  
🌐 GitHub: [Paul92150](https://github.com/Paul92150)
