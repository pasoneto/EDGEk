from vis import *
import matplotlib.pylab as plt
from tqdm import tqdm
import glob
import numpy as np
from scipy.cluster.hierarchy import linkage, leaves_list
import pandas as pd
import seaborn as sns

files = glob.glob(f"./data/test_exp5/motions_sliced/*.pkl")
files = glob.glob(f"./data/train_exp5/motions_sliced/*.pkl")

def get_corr(f, dims):
    df = np.load(f, allow_pickle=True)
    df = df.reshape(-1, 24, 3)#.numpy()
    if type(dims) == list: #Get average only if dims are more than one dimension
        print("Getting  dims")
        df = np.vstack([np.sum(df[:, l, dims], axis = 1) for l in range(24)])
    else:
        df = np.vstack([df[:, l, dims] for l in range(24)])
    correlations = np.corrcoef(df)
    assert correlations.shape == (24, 24)
    return correlations

df = np.mean([get_corr(l, [0, 1, 2]) for l in tqdm(files)], axis = 0)

list_of_names = ['root', 'rhip', 'lhip', 'belly', 'rknee', 'lknee', 'lchest', 'rankle', 'lankle', 'upchest', 
                 'rtoe', 'ltoe', 'neck', 'rclavicle', 'lclavicle', 'head', 'rshoulder', 'lshoulder', 
                 'relbow', 'lelbow', 'rwrist', 'lwrist', 'rhand', 'lhand']

df = pd.DataFrame(df)
df.columns = list_of_names
df.index = list_of_names

# Compute the linkage and optimal leaf order
linkage_matrix = linkage(df, method="average")
leaf_order = leaves_list(linkage_matrix)

# Reorder the DataFrame
ordered_matrix = df.iloc[leaf_order, leaf_order]

# Plot the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(ordered_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
plt.title("Similarity Matrix Heatmap with Clustered Correlations")
plt.show()

#AMASS (Dance)
#Left toe - right hand 0.27
#Left hand - Right hand 0.12

#AIST
#left toe - right hand 0.28
#Left hand - right hand 0.25
