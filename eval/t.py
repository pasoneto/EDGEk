import os
import pandas as pd
import numpy as np
from random import randint
from tqdm import tqdm

# Set paths
exp_number = 4
folders = os.listdir(f'../generated_dances/experiment{exp_number}/')

# List of marker names
list_of_names = ['root', 'rhip', 'lhip', 'belly', 'rknee', 'lknee', 'lchest', 'rankle', 'lankle', 'upchest', 
                 'rtoe', 'ltoe', 'neck', 'rclavicle', 'lclavicle', 'head', 'rshoulder', 'lshoulder', 
                 'relbow', 'lelbow', 'rwrist', 'lwrist', 'rhand', 'lhand']
# Generate marker names with appended numbers
markerNames = [f"{name}{i}" for name in list_of_names for i in range(1, 4)]
control = False
predicted_files = os.listdir(f'../generated_dances/experiment{exp_number}/{folders[3]}')

pred_file = predicted_files[0]
pred_path = os.path.join(f'../generated_dances/experiment{exp_number}/{folders[3]}', pred_file)

if control:
    index = randint(0, len(predicted_files) - 1)
    randomFile = predicted_files[index]
    randomFile = "_".join(randomFile.split("_")[2:])
    True_path = os.path.join('../data/test/motions_sliced', randomFile)
else:
    realName = "_".join(pred_file.split("_")[2:])
    True_path = os.path.join('../data/test/motions_sliced', realName)

True_data = np.load(True_path, allow_pickle=True).numpy()
pred_data = np.load(pred_path, allow_pickle=True)['full_pose']

# Placeholder class 'Dance' with attributes as per MATLAB equivalent
trueD = True_data
predE = pred_data
diffs = np.abs(trueD - predE)
m_for_markers = [np.mean(diffs[:, 3*l:3*(l+1)]) for l in range(len(list_of_names))]
mpe_case = pd.DataFrame([m_for_markers], columns=list_of_names)
np.mean(mpe_case)


