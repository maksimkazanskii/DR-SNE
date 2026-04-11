import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- Read CSV ---
df = pd.read_csv("output/timing/n.csv", header=None)
df = df.apply(pd.to_numeric, errors='coerce').dropna()

# --- Extract data ---
n = df.iloc[:, 0].values
mean = df.iloc[:, 1].values
std = df.iloc[:, 3].values

# --- Fit O(n^2) ---
def fit_scale(x, y):
    basis = x**2
    a = np.dot(y, basis) / np.dot(basis, basis)
    return a * basis

y_quad = fit_scale(n, mean)

# --- Style (NeurIPS-like minimal) ---
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#333333",
    "axes.labelcolor": "#333333",
    "xtick.color": "#333333",
    "ytick.color": "#333333",
    "grid.color": "#d9d9d9",
    "grid.linestyle": ":",
    "grid.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 12
})

# --- Plot ---
plt.figure(figsize=(7, 4.5))

# Mean
plt.plot(n, mean, color='steelblue', linewidth=1.6, label='Mean')

# Std band (lighter)
plt.fill_between(n, mean - std, mean + std,
                 color='steelblue', alpha=0.1)

# Std borders
plt.plot(n, mean - std, color='steelblue', linestyle='--', linewidth=1)
plt.plot(n, mean + std, color='steelblue', linestyle='--', linewidth=1,
         label=r'$\pm 1\,\mathrm{std}$')

# O(n^2) reference (subtle red)
plt.plot(n, y_quad,
         color='#d62728',
         linestyle='--',
         linewidth=1.2,
         alpha=0.7,
         label=r'$O(n^2)$')

# Grid
plt.grid(True)

# Labels (important for paper)
plt.xlabel('Number of samples (n)')
plt.ylabel('Runtime (seconds)')

# No title (NeurIPS style)
# plt.title(...)

# Legend
plt.legend(frameon=False)

# Layout + save
plt.tight_layout()
plt.savefig("output/time_plots/time_plot_neurips.png", dpi=300)
plt.close()