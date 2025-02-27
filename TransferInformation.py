# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 18:31:28 2025

@author: aaquiles
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import time
from tqdm import tqdm
import pickle


# Start timing
start = time.time()

# Calculate mutual information
def calc_MI(X, Y, bins='sturges'):
    # Remove NaN values
    valid_idx = ~np.isnan(X) & ~np.isnan(Y)
    X = X[valid_idx]
    Y = Y[valid_idx]
    
    binsX = np.histogram_bin_edges(X, bins=bins)
    binsY = np.histogram_bin_edges(Y, bins=bins)

    c_XY = np.histogram2d(X, Y, bins=[binsX, binsY])[0]
    c_X = np.histogram(X, bins=binsX)[0]
    c_Y = np.histogram(Y, bins=binsY)[0]

    H_X = shan_entropy(c_X)
    H_Y = shan_entropy(c_Y)
    H_XY = shan_entropy(c_XY)

    MI = H_X + H_Y - H_XY
    return MI
# Calculate Shannon entropy
def shan_entropy(c):
    c_normalized = np.nan_to_num(c / float(np.sum(c)))
    c_normalized = c_normalized[np.nonzero(c_normalized)]
    H = -np.sum(c_normalized * np.log2(c_normalized))
    return H

# Lag signals with edge handling
def lag_signal(signal, lag):
    if lag < 0:
        return np.pad(signal[:lag], (abs(lag), 0), mode='constant', constant_values=np.nan)
    elif lag > 0:
        return np.pad(signal[lag:], (0, lag), mode='constant', constant_values=np.nan)
    return signal

# Perform permutation test for statistical significance
def permutation_test(X, Y, lags, num_permutations=100):
    observed_mi = [calc_MI(lag_signal(X, lag), Y) for lag in lags]
    permuted_mi = np.zeros((num_permutations, len(lags)))

    for i in range(num_permutations):
        np.random.shuffle(Y)
        for j, lag in enumerate(lags):
            permuted_mi[i, j] = calc_MI(lag_signal(X, lag), Y)

    p_values = []
    for j in range(len(lags)):
        p = np.mean(permuted_mi[:, j] >= observed_mi[j])
        p_values.append(p)

    return np.array(observed_mi), np.array(p_values)


# Compute directionality with statistical testing
def compute_directionality(set_a, set_b, lags=np.arange(-50, 50), num_permutations=100):
    results = {
        "A_to_B": np.zeros((len(set_a), len(set_b), len(lags))),
        "B_to_A": np.zeros((len(set_b), len(set_a), len(lags))),
        "best_lags_A_to_B": np.zeros((len(set_a), len(set_b))),
        "best_lags_B_to_A": np.zeros((len(set_b), len(set_a))),
        "p_values_A_to_B": np.zeros((len(set_a), len(set_b), len(lags))),
        "p_values_B_to_A": np.zeros((len(set_b), len(set_a), len(lags)))
    }

    for i, a_signal in enumerate(tqdm(set_a, desc="Processing Set A")):
        for j, b_signal in enumerate(tqdm(set_b, desc=f"Processing Set B for A[{i}]", leave=False)):
            # A to B direction
            observed_mi_a_to_b, p_values_a_to_b = permutation_test(a_signal, b_signal, lags, num_permutations)
            results["A_to_B"][i, j, :] = observed_mi_a_to_b
            results["p_values_A_to_B"][i, j, :] = p_values_a_to_b
            results["best_lags_A_to_B"][i, j] = lags[np.argmax(observed_mi_a_to_b)]

            # B to A direction
            observed_mi_b_to_a, p_values_b_to_a = permutation_test(b_signal, a_signal, lags, num_permutations)
            results["B_to_A"][j, i, :] = observed_mi_b_to_a
            results["p_values_B_to_A"][j, i, :] = p_values_b_to_a
            results["best_lags_B_to_A"][j, i] = lags[np.argmax(observed_mi_b_to_a)]

    return results


# Visualization of MI vs. lag for one pair of signals
def plot_mi_lag(mi_values, lags, p_values=None, direction="A to B"):
    plt.figure(figsize=(8, 4))
    plt.plot(lags, mi_values, marker='o', label=f'{direction} MI')
    if p_values is not None:
        significant_lags = lags[p_values < 0.05]
        significant_mi = mi_values[p_values < 0.05]
        plt.scatter(significant_lags, significant_mi, color='red', label='Significant (p<0.05)')
    plt.axvline(lags[np.argmax(mi_values)], color='r', linestyle='--', label='Best Lag')
    plt.xlabel('Lag')
    plt.ylabel('Mutual Information (bits)')
    plt.title(f'MI vs. Lag ({direction})')
    plt.legend()
    plt.grid()
    plt.show()
    
    
with open('.pkl', 'wb') as fp:
    pickle.dump(results, fp)


### read files
with open('results_run_1.pkl', 'rb') as fp:
    results = pickle.load(fp)


# lags = np.arange(-20, 20)
# results = compute_directionality(ActBig, ActSmall, lags=lags, num_permutations=10)
plot_mi_lag(results['A_to_B'][0, 0, :], lags, results['p_values_A_to_B'][0, 0, :])


#%%

'''
           Run files sequentially 

'''


# Paths to your CSV files
files_to_run = [#("BigGroupD2.csv", "SmallGroupD2.csv"),
                #("BigGroupM4.csv", "SmallGroupM4.csv"),  ### male
                ("BigGroupE3.csv", "SmallGroupE3.csv"),
                #("BigGroupP2.csv", "SmallGroupP2.csv"),
                #("BigGroupL2.csv", "SmallGroupL2.csv"),
                #("BigGroupMO3.csv", "SmallGroupMO3.csv"),
                #("BigGroupV3.csv",  "SmallGroupV3.csv"),
                #("BigGroupM.csv", "SmallGroupM.csv"), ### multipara
                #("BigGroupL4.csv", "SmallGroupL4.csv")
                ]

# Output directory for results
output_dir = "results_output"
os.makedirs(output_dir, exist_ok=True)

# Loop through files and execute with tqdm
for idx, (big_path, small_path) in enumerate(tqdm(files_to_run, desc="Processing Files"), start=1):
    print(f"Processing files: {big_path} and {small_path}...")
    
    # Load the CSV files
    try:
        ActBig = np.loadtxt(big_path, delimiter = ';')
        ActSmall = np.loadtxt(small_path, delimiter = ';')
        # Drop rows with missing values
        # ActBig = np.dropna(ActBig)
        # ActSmall = np.dropna(ActSmall)
    except Exception as e:
        print(f"Error loading CSV files: {e}")
        continue

    # Define lags and run the computation
    lags = np.arange(-20, 20)
    results = compute_directionality(ActBig, ActSmall, lags=lags, num_permutations=10)

    # Save the results with a unique filename
    output_filename = os.path.join(output_dir, f"results_run_{idx}.pkl")
    with open(output_filename, "wb") as fp:
        pickle.dump(results, fp)
    print(f"Results saved to {output_filename}")
    
    # Optionally plot results for the first signal pair
    plot_mi_lag(results['A_to_B'][0, 0, :], lags, results['p_values_A_to_B'][0, 0, :])

print("All files processed.")


"""

    Spatial  temporal evaluation between transfer entropy values

"""

#%%


"""

          Shannon entropy to evaluate H between time series


"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy, mannwhitneyu

def shannon_entropy(time_series, bins=10):
    """

    Parameters:
        time_series (array-like): The input time series.
        bins (int): Number of bins to use for histogram discretization.

    Returns:
        float: The Shannon entropy of the time series.
    """
    hist, bin_edges = np.histogram(time_series, bins=bins, density=True)
    hist = hist[hist > 0]  # Remove zero probabilities
    return entropy(hist, base=2)

def compare_entropy_sets(set1, set2, bins=10):
    """
    Compare the entropy values of two sets of time series using a statistical test.

    Parameters:
        set1 (list of array-like): The first set of time series.
        set2 (list of array-like): The second set of time series.
        bins (int): Number of bins to use for histogram discretization.

    Returns:
        tuple: Median entropy values for both sets and p-value from the Mann-Whitney U test.
    """
    entropies1 = [shannon_entropy(ts, bins=bins) for ts in set1]
    entropies2 = [shannon_entropy(ts, bins=bins) for ts in set2]

    stat, p_value = mannwhitneyu(entropies1, entropies2, alternative='two-sided')

    return np.median(entropies1), np.median(entropies2), p_value

def plot_entropy_distribution(entropies1, entropies2, labels):

    plt.figure(figsize=(10, 6))
    plt.boxplot(entropies1, )
    plt.boxplot(entropies2, )
    # plt.hist(entropies1, bins=20, alpha=0.7, label=labels[0], color='blue', edgecolor='black')
    # plt.hist(entropies2, bins=20, alpha=0.7, label=labels[1], color='orange', edgecolor='black')
    plt.xlabel('Entropy')
    plt.ylabel('Frequency')
    plt.title('Entropy Distribution Estrus')
    plt.legend()
    plt.grid(True)
    plt.show()


Act1 = np.loadtxt("BigGroupE3.csv",delimiter=';')
Act2 = np.loadtxt("SmallGroupE3.csv",delimiter=';')


# Compare entropy between the two sets
median1, median2, p_value = compare_entropy_sets(Act1, Act2)
print(f"Median entropy of Set 1: {median1}")
print(f"Median entropy of Set 2: {median2}")
print(f"P-value from Mann-Whitney U test: {p_value}")

# Plot entropy distributions
entropies1 = [shannon_entropy(ts) for ts in Act1]
entropies2 = [shannon_entropy(ts) for ts in Act2]
plot_entropy_distribution(entropies1, entropies2, labels=("Set 1", "Set 2"))


entropies1 = np.array(entropies1)
entropies2 = np.array(entropies2)

#%%%

df = pd.read_csv('EntropyValues.csv', delimiter =';')

sns.boxplot(data=df, x="Conditions", y="Median", hue="Type")

#%%


#### Spatial temporal patterns from TE 


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

### read files
with open('results_run_D2.pkl', 'rb') as fp:
    results = pickle.load(fp)


# Generate synthetic data for demonstration (replace this with your TE data)
np.random.seed(42)
T = 50  # Number of time windows
TE_matrix = np.random.rand(N, N, T) * 0.1  # Simulated small TE values
TE_matrix += np.random.rand(N, N, 1) * 0.3  # Add stronger connections


TE_matrix = te_values
N = 3  # Number of neurons
# Set a threshold for significant TE
TE_threshold = 0.2
TE_matrix_thresh = np.where(TE_matrix > TE_threshold, TE_matrix, 0)

# Spatial Pattern: Average TE per neuron (input and output)
average_inflow = TE_matrix_thresh.sum(axis=0).mean(axis=1)  # Average incoming TE
average_outflow = TE_matrix_thresh.sum(axis=1).mean(axis=1)  # Average outgoing TE

# Temporal Pattern: Total TE over time
total_TE_over_time = TE_matrix_thresh.sum(axis=(0, 1))  # Sum of all TE over time

# Plot Results
plt.figure(figsize=(12, 8))

# Plot spatial inflow and outflow patterns
plt.subplot(2, 1, 1)
x = np.arange(1, N + 1)
plt.bar(x - 0.2, average_inflow, width=0.4, label='Inflow', color='skyblue')
plt.bar(x + 0.2, average_outflow, width=0.4, label='Outflow', color='orange')
plt.xlabel('Neuron Index')
plt.ylabel('Average TE')
plt.title('Spatial Patterns of Transfer Entropy')
plt.legend()

# Plot temporal patterns
plt.subplot(2, 1, 2)
plt.plot(total_TE_over_time, label='Total TE Over Time', color='purple')
plt.xlabel('Time Window')
plt.ylabel('Total TE')
plt.title('Temporal Patterns of Transfer Entropy')
plt.legend()

plt.tight_layout()
plt.show()

# Visualize Connectivity Matrix for a Specific Time
time_idx = 10  # Example time window
plt.figure(figsize=(8, 6))
sns.heatmap(TE_matrix_thresh[:, :, time_idx], annot=False, cmap='viridis')
plt.title(f'Transfer Entropy Connectivity (Time Window {time_idx})')
plt.xlabel('Neuron j')
plt.ylabel('Neuron i')
plt.show()


#%%

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm 
import os
import pickle
"""
Transfer Entropy (TE) measures the information transfer from one time series (source) to another (target). TE is defined using probability distributions. For continuous-valued signals like time series, the probability distributions are not readily available, because there are infinitely many possible values.

Discretization allows us to:

    Approximate probability distributions: By dividing the continuous signal into discrete bins (e.g., 10 equally spaced intervals), we can estimate the probability of a value falling into each bin.
    Simplify computation: Working with discrete values (bins) makes it feasible to compute joint probabilities and conditional probabilities needed for TE.


    ## , using np.genfromtxt with missing_values could handle missing values better.


"""





def compute_te(source_signal, target_signal, lag=1, n_bins=10):
    """
    Computes Transfer Entropy (TE) between a source and target signal using binned probability distributions.

    Parameters:
        source_signal (ndarray): Source signal (1D array).
        target_signal (ndarray): Target signal (1D array).
        lag (int): Lag between the source and target signals for TE computation.
        n_bins (int): Number of bins for probability distributions.

    Returns:
        float: Transfer Entropy (TE) value.
        
    """
    # Ensure lag does not exceed signal length
    if lag >= len(source_signal):
        raise ValueError("Lag must be smaller than the length of the signals.")

    # Prepare lagged signals
    X = source_signal[:-lag]        # Source at time t-lag
    Y = target_signal[lag:]         # Target at time t
    Y_prev = target_signal[:-lag]   # Target at time t-lag

    # Discretize signals into bins
    X_binned = np.digitize(X, np.histogram(X, bins=n_bins)[1]) - 1
    Y_binned = np.digitize(Y, np.histogram(Y, bins=n_bins)[1]) - 1
    Y_prev_binned = np.digitize(Y_prev, np.histogram(Y_prev, bins=n_bins)[1]) - 1

    # Compute joint and marginal probabilities
    p_xy = np.histogram2d(X_binned, Y_binned, bins=n_bins)[0] / len(X_binned)
    p_xyp = np.histogramdd((X_binned, Y_binned, Y_prev_binned), bins=(n_bins, n_bins, n_bins))[0] / len(X_binned)
    p_yp = np.histogram2d(Y_binned, Y_prev_binned, bins=n_bins)[0] / len(X_binned)
    p_y = np.histogram(Y_binned, bins=n_bins)[0] / len(X_binned)

    # Avoid log(0) by replacing zeros with a small value
    epsilon = 1e-10
    p_xy += epsilon
    p_xyp += epsilon
    p_yp += epsilon
    p_y += epsilon

    # Compute TE as the sum of conditional mutual information terms
    te = np.sum(p_xyp * np.log((p_xyp * p_y) / (p_xy * p_yp)))

    return te

def compute_te_between_datasets(set1, set2, window_size=500, lag=100, n_bins=10, n_permutations=5, alpha=0.05):
    """
    Computes TE between two datasets with statistical significance testing.

    Parameters:
        set1 (ndarray): Dataset 1 (e.g., shape (81, 1000)).
        set2 (ndarray): Dataset 2 (e.g., shape (12, 1000)).
        window_size (int): Size of the sliding window.
        lag (int): Lag for TE computation.
        n_bins (int): Number of bins for TE computation.
        n_permutations (int): Number of permutations for significance testing.
        alpha (float): Significance level.

    Returns:
        te_values (ndarray): TE values (shape: (n_set1, n_set2, T - window_size)).
        significant_te (ndarray): Binary array indicating significance (same shape as `te_values`).
    """
    n_set1, T = set1.shape
    n_set2 = set2.shape[0]
    
    # Initialize TE values and significance arrays
    te_values = np.zeros((n_set1, n_set2, T - window_size))
    significant_te = np.zeros_like(te_values)
    
    # Sliding window computation with tqdm for progress bar
    for t in tqdm(range(T - window_size), desc="Computing TE", unit="time point"):
        for source_idx in range(n_set1):
            for target_idx in range(n_set2):
                # Get source and target signals in the current window
                source_signal = set1[source_idx, t:t+window_size]
                target_signal = set2[target_idx, t:t+window_size]
                
                # Compute true TE
                true_te = compute_te(source_signal, target_signal, lag=lag, n_bins=n_bins)
                
                # Permutation test
                permuted_tes = []
                for _ in range(n_permutations):
                    shuffled_target = np.random.permutation(target_signal)
                    permuted_te = compute_te(source_signal, shuffled_target, lag=lag, n_bins=n_bins)
                    permuted_tes.append(permuted_te)
                
                # Determine significance
                permuted_tes = np.array(permuted_tes)
                threshold = np.percentile(permuted_tes, 100 * (1 - alpha))  # Upper alpha quantile
                te_values[source_idx, target_idx, t] = true_te
                significant_te[source_idx, target_idx, t] = 1 if true_te > threshold else 0

    return te_values, significant_te


# np.random.seed(42)
# set1 = np.random.rand(3, 1000)  # Dataset 1 (81 signals, 1000 time points)
# set2 = np.random.rand(7, 1000)  # Dataset 2 (12 signals, 1000 time points)

# Compute TE with significance testing
window_size = 500
lag = 100
te_values, significant_te = compute_te_between_datasets(set1, set2, window_size=window_size, lag=lag)

# Visualization for the first source-target pair
time_range = range(set1.shape[1] - window_size)
plt.figure(figsize=(12, 6))
plt.plot(time_range, te_values[0, 0, :], label="TE (Source 0 → Target 0)")
plt.fill_between(time_range, 0, te_values[0, 0, :], where=significant_te[0, 0, :].astype(bool),
                 color='red', alpha=0.3, label="Significant TE")
plt.xlabel('Time')
plt.ylabel('TE')
plt.title('TE Between Source 0 and Target 0 with Significance')
plt.legend()
plt.grid(True)
plt.show()


###### compute several files 



# Paths to your CSV files
files_to_run = [("BigGroupE3.csv", "SmallGroupE3.csv")]

# Output directory for results
output_dir = "results_output"
os.makedirs(output_dir, exist_ok=True)

# Loop through files and execute with tqdm
for idx, (big_path, small_path) in enumerate(tqdm(files_to_run, desc="Processing Files"), start=1):
    print(f"Processing files: {big_path} and {small_path}...")
    
    # Load the CSV files
    try:
        # Use np.genfromtxt to handle potential missing values more robustly
        ActBig = np.genfromtxt(big_path, delimiter=';', missing_values='', filling_values=np.nan)
        ActSmall = np.genfromtxt(small_path, delimiter=';', missing_values='', filling_values=np.nan)
        
        # Drop rows with missing values (optional, depends on data quality)
        ActBig = ActBig[~np.isnan(ActBig).any(axis=1)]
        ActSmall = ActSmall[~np.isnan(ActSmall).any(axis=1)]
        
    except Exception as e:
        print(f"Error loading CSV files: {e}")
        continue

    # Define lags and run the computation
    lags = np.arange(-20, 20)
    results = compute_te_between_datasets(ActBig, ActSmall, window_size=150, lag=50, n_bins=20)   ## change bins 

    # Save the results with a unique filename
    output_filename = os.path.join(output_dir, f"results_run_{idx}.pkl")
    with open(output_filename, "wb") as fp:
        pickle.dump(results, fp)
    print(f"Results saved to {output_filename}")
    
    # Optionally plot results for the first source-target pair
    plt.figure(figsize=(12, 6))
    plt.plot(np.arange(len(results[0][0])), results[0][0, 0, :], label="TE (Source 0 → Target 0)")
    plt.xlabel('Time')
    plt.ylabel('TE')
    plt.title(f'Transfer Entropy (Source 0 → Target 0) for run {idx}')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f"te_plot_run_{idx}.png"))
    plt.close()

print("All files processed.")
