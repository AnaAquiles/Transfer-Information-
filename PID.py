# -*- coding: utf-8 -*-
"""
Created on Thu Apr 2025

@author: aaquiles
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy

"""
          Computing PID only for lag time points

   https://www.cell.com/current-biology/fulltext/S0960-9822(24)00320-8
"""

def mutual_information(X, Y):
    """Compute mutual information between two variables using histogram-based entropy."""

    binsY = np.histogram_bin_edges(Y, bins='sturges', range=(0, 5))
    binsX = np.histogram_bin_edges(X, bins='sturges', range=(0, 5))

    c_XY = np.histogram2d(X, Y, bins=[binsX, binsY])[0]
    c_X = np.histogram(X, binsX)[0]
    c_Y = np.histogram(Y, binsY)[0]

    H_X = shan_entropy(c_X)
    H_Y = shan_entropy(c_Y)
    H_XY = shan_entropy(c_XY)

    mi = H_X + H_Y - H_XY
    return mi

def shan_entropy(c):
    c_normalized = np.nan_to_num(c / float(np.sum(c)))
    c_normalized = c_normalized[np.nonzero(c_normalized)]
    H = -sum(c_normalized * np.log2(c_normalized))
    return H

# Function to compute a simple PID decomposition approximation for multiple sources
def pid_decomposition(sources, target):
    """Approximate unique, redundant, and synergistic information components for multiple sources."""
    num_sources = len(sources)
    mi_values = [mutual_information(src, target) for src in sources]
    mi_joint = mutual_information(np.mean(np.column_stack(sources), axis=1), target)

    redundancy = min(mi_values)  # Approximate redundancy as the minimum MI
    synergy = mi_joint - max(mi_values)  # Synergy as the difference from the strongest MI
    unique_info = [mi - redundancy for mi in mi_values]  # Unique information for each source

    return np.array(unique_info + [redundancy, synergy])  # Convert to NumPy array

# Function to compute PID across different time lags
def compute_pid_lags(sources, target, lags):
    """Compute PID for different time lags."""
    pid_matrix = np.zeros((len(lags), len(sources) + 2))  # Store results

    for i, lag in enumerate(lags):
        shifted_target = np.roll(target, lag)  # Shift target by lag
        pid_matrix[i] = pid_decomposition(sources, shifted_target)

    return pid_matrix

# Simulated LFP time series data (3 sources now!)
np.random.seed(42)
N = 1000  # Number of samples
X1 = np.sin(np.linspace(0, 10, N)) + 0.1 * np.random.randn(N)  # Source 1
X2 = np.cos(np.linspace(0, 10, N)) + 0.1 * np.random.randn(N)  # Source 2
X3 = np.sin(2 * np.linspace(0, 10, N)) + 0.1 * np.random.randn(N)  # Source 3
Z = X1 * X2 * X3 + 0.1 * np.random.randn(N)  # Target with interaction

sources = [X1, X2, X3]

# Define time lags (e.g., -50 to +50 samples)
lags = np.arange(-50, 51)

# Compute PID across time lags
pid_lagged = compute_pid_lags(sources, Z, lags)

# Visualization as a heatmap (like Lemke et al.)
labels = [f"Unique X{i+1}" for i in range(len(sources))] + ["Redundancy", "Synergy"]
fig, axes = plt.subplots(1, len(labels), figsize=(15, 5), sharey=True)

for i, (ax, label) in enumerate(zip(axes, labels)):
    im = ax.imshow(pid_lagged[:, i].reshape(-1, 1), cmap="coolwarm", aspect="auto", extent=[0, 1, lags[0], lags[-1]])
    ax.set_title(label)
    ax.set_xticks([])
    ax.set_ylabel("Lag (samples)")
    fig.colorbar(im, ax=ax)

plt.suptitle("PID Decomposition Across Time Lags")
plt.show()
