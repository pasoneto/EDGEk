from vis import *
import matplotlib.pylab as plt
from tqdm import tqdm

files = glob.glob(f"./data/train_exp3/motions_sliced/*.pkl")

def get_corr(f, dims):
    df = np.load(f, allow_pickle=True)
    df = df.reshape(-1, 24, 3).numpy()
    if type(dims) == list: #Get average only if dims are more than one dimension
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

def heatmap2d(arr: np.ndarray):
    plt.imshow(arr, cmap='viridis')
#    plt.xticks(list_of_names)
#    plt.yticks(list_of_names)
    plt.colorbar()
    plt.show()

for k in range(24):
    for j in range(24):
        if k != j:
            if np.abs(df[k, j]) < 0.2:
                print(f"{list_of_names[k]}, {list_of_names[j]}")

heatmap2d(df)

##All dimensions
#AMASS
##lwrist, rhand
##rhand, lhand

#AIST
##lelbow, rwrist
##lelbow, rhand

##Dimension 0          Dimension 1            Dimension 2 (up-down)
#AMASS
##lwrist, rhand        lwrist, rhand          belly, rtoe
##rhand, lhand         rhand, lhand           rtoe, lelbow 

#AIST
##Dimension 0          Dimension 1            Dimension 2
##lwrist, rhand        lwrist, rhand,         belly, rtoe
##rhand,  lhand        rhand, lhand,          rtoe, lelbow 

