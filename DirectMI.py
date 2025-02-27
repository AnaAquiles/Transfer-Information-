# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 22:01:27 2024

@author: aaquiles
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import pickle

start = time.time()

def calc_MI(X,Y):
    
    binsY = np.histogram_bin_edges(Y,bins = 'sturges', range =(0,5))
    binsX = np.histogram_bin_edges(X,bins = 'sturges', range =(0,5))
    
    c_XY = np.histogram2d(X,Y,binsX)[0]
    c_X = np.histogram(X,binsX)[0]
    c_Y = np.histogram(Y,binsY)[0]
 
    H_X = shan_entropy(c_X)
    H_Y = shan_entropy(c_Y)
    H_XY = shan_entropy(c_XY)
 
    MI = H_X + H_Y - H_XY
    return MI
 
def shan_entropy(c):
    c_normalized = np.nan_to_num(c / float(np.sum(c)))
    c_normalized = c_normalized[np.nonzero(c_normalized)]
    H = -sum(c_normalized* np.log2(c_normalized))  
    return H



def compute_directionality(set_a, set_b, lags=np.arange(-5, 5)):
    """
    Compute directionality between two signal sets (Set A -> Set B and Set B -> Set A).
    Returns the directionality flow and best lags for each signal pair.
    """
    results = {
        "A_to_B": np.zeros((len(set_a), len(set_b), len(lags))),
        "B_to_A": np.zeros((len(set_b), len(set_a), len(lags))),
        "best_lags_A_to_B": np.zeros((len(set_a), len(set_b))),
        "best_lags_B_to_A": np.zeros((len(set_b), len(set_a))),
    }
    
    for i, a_signal in enumerate(set_a):
        for j, b_signal in enumerate(set_b):
            mi_a_to_b = []
            mi_b_to_a = []
            for lag in lags:
                if lag < 0:
                    lagged_b = np.roll(b_signal, lag)
                    lagged_a = a_signal
                else:
                    lagged_a = np.roll(a_signal, lag)
                    lagged_b = b_signal
                mi_a_to_b.append(calc_MI(lagged_a, lagged_b))
                mi_b_to_a.append(calc_MI(lagged_b, lagged_a))
            results["A_to_B"][i, j, :] = mi_a_to_b
            results["B_to_A"][j, i, :] = mi_b_to_a
            results["best_lags_A_to_B"][i, j] = lags[np.argmax(mi_a_to_b)]
            results["best_lags_B_to_A"][j, i] = lags[np.argmax(mi_b_to_a)]
    
    return results

# Generate synthetic signals
# np.random.seed(42)
# t = np.linspace(0, 10, 1000)

# # Signal sets
# set_A = [np.sin(2 * np.pi * t) + 0.1 * np.random.randn(1000),  # A1
#          np.cos(2 * np.pi * t) + 0.1 * np.random.randn(1000)]  # A2

# set_B = [np.roll(set_A[0], 5) + 0.1 * np.random.randn(1000),   # B1
#          np.roll(set_A[1], -10) + 0.1 * np.random.randn(1000)] # B2

ActBig  = np.loadtxt("BigGroupL2.csv",delimiter=';')   
ActSmall  = np.loadtxt("SmallGroupL2.csv",delimiter=';')   


# Compute directionality
lagsf = np.arange(-60, 60)
resultsf = compute_directionality(ActBig, ActSmall, lagsf)


with open('lactL2.pkl', 'wb') as fp:
    pickle.dump(resultsf, fp)

# Aggregate and plot results
A_to_B = resultsf["A_to_B"]
bestlagsAb = resultsf["best_lags_A_to_B"]
B_to_A = resultsf["B_to_A"]

Direct1 = A_to_B
Direct2 = B_to_A


# Plot example pair (A1 -> B1)
plt.figure(figsize=(10, 6))
plt.plot(lagsf, A_to_B[0, 0, :], label="MI(A1 -> B1)", color="blue")
plt.plot(lagsf, B_to_A[0, 0, :], label="MI(B1 -> A1)", color="red")
plt.axvline(0, color='black', linestyle='--', label='Zero lag')
plt.xlabel("Lag (time steps)")
plt.ylabel("Mutual Information")
plt.title("Directionality Flow: A1 <-> B1")
plt.legend()
plt.show()

# Print best lags for all pairs
print("Best Lags (A -> B):")
print(resultsf["best_lags_A_to_B"])
print("Best Lags (B -> A):")
print(resultsf["best_lags_B_to_A"])

print('It took', time.time()-start, 'seconds.')

#%%


"""
       MI directionality thresholded by 
       
       Max - 2SD and/ord  Mean + 2SD

"""

def BinMat (Mat, maximum = True):
    MaTh = Mat
    Mean = np.mean(MaTh, axis = 2)   # Act
    Std = np.std(MaTh, axis = 2)
    
    
    ThMI = Mean + 2*Std
    ThMI_expanded = ThMI[:, :, np.newaxis]  # we need to add and extra axis to use as a mask to filter every matrix 
    
    MaTh[MaTh < ThMI_expanded]= 0
    MaTh[MaTh > ThMI_expanded]= 1
    
    if not maximum:
        return MaTh
    
    MaxMI = np.max(Mat, axis = 2)   # Act
    StdMI = np.std(Mat, axis = 2)
    
    
    ThMI1 = MaxMI - 2*StdMI
    ThMI_expanded1 = ThMI1[:, :, np.newaxis]  # we need to add and extra axis to use as a mask to filter every matrix 
    
    Mat[Mat < ThMI_expanded1]= 0
    Mat[Mat > ThMI_expanded1]= 1
    
    return Mat

MatTh = BinMat(np.abs(Direct2), maximum = True)


"""
        Build the vectors among every time point 
       
"""


MatchResh = MatTh.reshape((1026,120))

#%%

##### make a GIF of every frame only to chek it out


import os
import numpy as np
import matplotlib.pyplot as plt
import imageio
from scipy import stats


plt.style.use('ggplot')


filenames = []
for i in range(len(MatTh[0,0,:])):
    plt.imshow(MatTh[:,:,i], aspect = 'auto')
    plt.title('Direct 2 (Spikes to Swing)')
    plt.xlabel('Swing cells')
    plt.ylabel('Spike type cells')
    plt.box(False)
    plt.grid(False)
    filename = f'{i}.png'
    filenames.append(filename)
    plt.savefig(filename)
    plt.close()
    
with imageio.get_writer('Lactating2HomoD2.gif', mode='I') as writer:
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)
        
for filename in set(filenames):
    os.remove(filename)        



#%%


"""
           plot the MUTUAL INFORMATION DIRECTION
"""

from scipy.stats import gaussian_kde

with open('results_run_1.pkl', 'rb') as fp:
    results = pickle.load(fp)


CastBA = np.mean(resultsf["B_to_A"], axis = 2)
CastAB = np.mean(resultsf["A_to_B"], axis = 2)


# Fit KDE for each dataset
kde1 = gaussian_kde(CastAB.reshape(820))
kde2 = gaussian_kde(CastAB.reshape(820))
# kde3 = gaussian_kde(CastAB[:,2])
# kde4 = gaussian_kde(CastAB[:,3])
# kde5 = gaussian_kde(CastAB[:,4])


# Define a common range for evaluation
x = np.linspace(0, 1, 10)  # A range of values covering both datasets

# Evaluate the KDEs
kde1_values = kde1(x)
kde2_values = kde2(x)
kde3_values = kde3(x)
kde4_values = kde4(x)
kde5_values = kde5(x)

# Normalize the KDEs so their integral equals 1
kde1_values_normalized = kde1_values / np.sum(kde1_values)
kde2_values_normalized = kde2_values / np.sum(kde2_values)
kde3_values_normalized = kde3_values / np.sum(kde3_values)
kde4_values_normalized = kde4_values / np.sum(kde4_values)
kde5_values_normalized = kde5_values / np.sum(kde5_values)

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(x, kde1_values_normalized, label="KDE 1 (slow to fast)", color="black", alpha = 0.5)
plt.plot(x, kde2_values_normalized, label="KDE 2 (fast to slow)", color="yellow", alpha = 0.5)
# plt.plot(x, kde3_values_normalized, label="KDE 3 (normalized)", color="blue", alpha = 0.5)
# plt.plot(x, kde4_values_normalized, label="KDE 4 (normalized)", color="green", alpha = 0.5)
# plt.plot(x, kde5_values_normalized, label="KDE 5 (normalized)", color="black", alpha = 0.8)
plt.title("Slow and Fast population Male Castrated")
plt.xlabel("Value")
plt.ylabel("Density")
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
plt.subplot(121)
plt.title('Big group to small')
plt.imshow(CastAB, aspect = 'auto', cmap = 'viridis', vmin = 0, vmax = 2)
plt.grid(False)
plt.colorbar()
plt.subplot(122)
plt.title('Small group to Big')
plt.imshow(CastBA.T, aspect = 'auto', cmap = 'viridis', vmin = 0, vmax = 2)
plt.grid(False)
plt.colorbar()

#%%

Df = pd.read_csv('ElisaResults2024-1.csv', delimiter = ';')
Df = Df[Df['TSH2'] < 25]
sns.set_theme(style="ticks")



sns.relplot(
    data=Df, x="Time2", y="TSH2", col="Mice2",
    hue="Mice2", kind="line",linewidth=5)

# g = sns.relplot(
#     data=flights,
#     x="Time2", y="Mice", col="year", hue="year",
#     kind="line", palette="crest", linewidth=4, zorder=5,
#     col_wrap=3, height=2, aspect=1.5, legend=False, )