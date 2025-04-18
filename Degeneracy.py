import itertools
import numpy as np
from sklearn.metrics import mutual_info_score

"""
   Code based on https://www.ncbi.nlm.nih.gov/pmc/articles/PMC15929/ 
"""


def calculate_mutual_information(X_subset, O):
    
    if X_subset.ndim == 1:
        # If the subset is a single feature, mutual_info_score can be applied directly
        return mutual_info_score(X_subset, O)
    else:
        # For multiple features, discretize and compute mutual information for each feature, then sum
        mi = 0
        for i in range(X_subset.shape[1]):
            mi += mutual_info_score(X_subset[:, i], O)
        return mi

def degeneracy(X, O):
  
    n = X.shape[1]
    MIP_X_O = calculate_mutual_information(X, O)
    
    degeneracy_sum = 0
    for k in range(1, n + 1):
        subsets = itertools.combinations(range(n), k)
        MIP_subsets = []
        for subset in subsets:
            X_subset = X[:, subset]
            MIP_subsets.append(calculate_mutual_information(X_subset, O))
        avg_MIP_subsets = np.mean(MIP_subsets)
        degeneracy_sum += (k / n) * MIP_X_O - avg_MIP_subsets
    
    return degeneracy_sum

# Example usage:
# X is a 2D NumPy array with shape (time series(x,t))
# O is a 1D NumPy array with shape (output) need to specify a chunk of your data 
