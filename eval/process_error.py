import os
import pandas as pd
import numpy as np
from random import randint
from tqdm import tqdm
import sys
import torch

sys.path.insert(1, '/Users/pdealcan/Documents/github/edge_redo/EDGEk/')

from data.accel_extraction_funcs import remove_foot_contact_and_fk

# Set paths
exp_number = 5
folders = os.listdir(f'../generated_dances/experiment{exp_number}/')

# List of marker names
list_of_names = ['root', 'rhip', 'lhip', 'belly', 'rknee', 'lknee', 'lchest', 'rankle', 'lankle', 'upchest', 
                 'rtoe', 'ltoe', 'neck', 'rclavicle', 'lclavicle', 'head', 'rshoulder', 'lshoulder', 
                 'relbow', 'lelbow', 'rwrist', 'lwrist', 'rhand', 'lhand']

# Generate marker names with appended numbers
markerNames = [f"{name}{i}" for name in list_of_names for i in range(1, 4)]
controls = [True, False]
for l in folders:
    predicted_files = os.listdir(f'../generated_dances/experiment{exp_number}/{l}')
    for control in controls:
        all_files = []
        for k in tqdm(range(len(predicted_files)), desc="Processing files", unit="file"):
            pred_file = predicted_files[k]
            pred_path = os.path.join(f'../generated_dances/experiment{exp_number}/{l}', pred_file)
            if control:
                index = randint(0, len(predicted_files) - 1)
                randomFile = predicted_files[index]
                randomFile = "_".join(randomFile.split("_")[2:])
                True_path = os.path.join(f'../data/test_exp{exp_number}/motions_sliced', randomFile)
            else:
                realName = pred_file
                realName = "_".join(pred_file.split("_")[2:])
                True_path = os.path.join(f'../data/test_exp{exp_number}/motions_sliced', realName)
#                assert pred_file == realName

            True_data = np.load(True_path, allow_pickle=True)#.numpy()
            pred_data = np.load(pred_path, allow_pickle=True)['full_pose']
            
            if True_data.shape[1] == 151:
                True_data = remove_foot_contact_and_fk(torch.from_numpy(True_data))
                True_data = True_data.reshape(-1, 24*3).numpy()
                pred_data = pred_data.reshape(-1, 24*3)

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
            gtc_case['file'] = pred_file
            gtc_case['condition'] = "control" if control else "experiment"

            objective_measure_case = pd.concat([mpe_case, gtc_case], axis=1)
            all_files.append(objective_measure_case)

        # Combine all files into a single DataFrame
        combined_table = pd.concat(all_files, ignore_index=True)
        combined_table['experiment_run'] = l

        # Set output file path based on control flag
        output_filename = f"./eval_data/objective_eval_exp{exp_number}/objective_measure_control_{l}.csv" if control else \
                          f"./eval_data/objective_eval_exp{exp_number}/objective_measure_experimental_{l}.csv"
        combined_table.to_csv(output_filename, index=False)
        print(f"Ended folder: {l}")
