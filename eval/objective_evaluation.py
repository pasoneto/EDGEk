import os
import pandas as pd
import numpy as np
from random import randint
from tqdm import tqdm

# Set paths
predicted_files = os.listdir('../generated_dances/')
true_files = os.listdir('../data/test/motions_sliced/')

# List of marker names
list_of_names = ['root', 'rhip', 'lhip', 'belly', 'rknee', 'lknee', 'lchest', 'rankle', 'lankle', 'upchest', 
                 'rtoe', 'ltoe', 'neck', 'rclavicle', 'lclavicle', 'head', 'rshoulder', 'lshoulder', 
                 'relbow', 'lelbow', 'rwrist', 'lwrist', 'rhand', 'lhand']

# Generate marker names with appended numbers
markerNames = [f"{name}{i}" for name in list_of_names for i in range(1, 4)]

control = False
all_files = []
for k in tqdm(range(len(predicted_files)), desc="Processing files", unit="file"):
    nameChosen = predicted_files[k]
    if control:
        index = randint(0, len(predicted_files) - 1)
        nameChosenRandom = predicted_files[index]
        nameChosenRandomReal = "_".join(nameChosenRandom.split("_")[2:])
        True_path = os.path.join('../data/test/motions_sliced', nameChosenRandomReal)
        pred_path = os.path.join('../generated_dances', nameChosenRandom)
    else:
        realName = "_".join(nameChosen.split("_")[2:])
        True_path = os.path.join('../data/test/motions_sliced', realName)
        pred_path = os.path.join('../generated_dances', nameChosen)
    
    True_data = np.load(True_path, allow_pickle=True).numpy()
    pred_data = np.load(pred_path, allow_pickle=True)['full_pose']

    # Placeholder class 'Dance' with attributes as per MATLAB equivalent
    trueD = True_data
    predE = pred_data
    diffs = np.abs(trueD - predE)

    m_for_markers = [np.mean(diffs[:, 3*l:3*(l+1)]) for l in range(len(list_of_names))]
    mpe_case = pd.DataFrame([m_for_markers], columns=list_of_names)

    first_dimension_indexes = np.arange(0, trueD.shape[1], 3)
    second_dimension_indexes = np.arange(1, trueD.shape[1], 3)
    third_dimension_indexes = np.arange(2, trueD.shape[1], 3)

    first_dimension = np.corrcoef(trueD[:, first_dimension_indexes].mean(axis=1),
                                  predE[:, first_dimension_indexes].mean(axis=1))[0, 1]
    second_dimension = np.corrcoef(trueD[:, second_dimension_indexes].mean(axis=1),
                                   predE[:, second_dimension_indexes].mean(axis=1))[0, 1]
    third_dimension = np.corrcoef(trueD[:, third_dimension_indexes].mean(axis=1),
                                  predE[:, third_dimension_indexes].mean(axis=1))[0, 1]
    
    gtc_case = pd.DataFrame([[first_dimension, second_dimension, third_dimension]], 
                            columns=["gtc_first", "gtc_second", "gtc_third"])
    gtc_case['file'] = nameChosen
    gtc_case['condition'] = "control" if control else "experiment"

    objective_measure_case = pd.concat([mpe_case, gtc_case], axis=1)
    all_files.append(objective_measure_case)

# Combine all files into a single DataFrame
combined_table = pd.concat(all_files, ignore_index=True)

# Set output file path based on control flag
output_filename = "./eval_data/objective_eval/objective_measure_control.csv" if control else \
                  "./eval_data/objective_eval/objective_measure_experimental.csv"
combined_table.to_csv(output_filename, index=False)
