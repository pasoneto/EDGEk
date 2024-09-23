import sys
sys.path.insert(1, '/Users/pdealcan/Documents/github/EDGEk/')

import numpy as np
import torch
import os
import pandas as pd
from pytorch3d.transforms import (axis_angle_to_matrix)
import json
import pickle as pkl
from dataset.dance_dataset import *
from vis import SMPLSkeleton, smplToPosition, create_middle_marker, differentiate_fast
from pytorch3d.transforms import (axis_angle_to_quaternion, quaternion_apply,
                                  quaternion_multiply, quaternion_to_axis_angle, RotateAxisAngle)
from dataset.quaternion import ax_from_6v, quat_slerp

from scipy.interpolate import interp1d

def interp(df, sr, newfreq):
    t1 = np.arange(0, df.shape[0]) / sr
    t2 = np.arange(0, t1[-1], 1/newfreq)
    interpolator = interp1d(t1, df, axis=0, kind='linear', fill_value="extrapolate")
    interpolated_data = interpolator(t2)
    return interpolated_data

def slicer(df, rows, i):
    df_slice = {}
    df_slice['trans'] = df['trans'][i * rows : (i + 1) * rows, :]
    df_slice['poses'] = df['poses'][i * rows : (i + 1) * rows, :]
    df_slice['scaling'] = 1
    return df_slice

path = "/Users/pdealcan/Documents/github/data/CoE/accel/amass/amass_full/DanceDB/"
dirOut2 = "./eval/eval_data/positions_amass/"
datasets = os.listdir(path)

seconds = 10
newFreq = 15
folders = os.listdir(f"{path}")
for k in folders:
    if k != ".DS_Store":
        files = os.listdir(f"{path}/{k}")
        for j in files:
            if j != "shape.npz":
                try:
                    cFile = f"{path}/{k}/{j}"

                    df = dict(np.load(cFile))
                    sr = df['mocap_framerate'] 

                    df['scaling'] = 1

                    #Resample
                    df['trans'] = interp(df['trans'], sr, newFreq)
                    df['poses'] = interp(df['poses'], sr, newFreq)
                    sr = newFreq 

                    #Slicing
                    n_frames = len(df['trans'])
                    rows = int(sr*seconds)
                    num_chunks = int(n_frames / rows) #int() always rounds down, therefore we never get an unequal sample size
                    for i in range(num_chunks):
                        df_sliced = slicer(df, rows, i)
                        positions, rotations = smplToPosition(df_sliced['trans'], df_sliced['poses'], df_sliced['scaling'], aist = False)

                        positions = positions[0]
                        angles = quaternion_to_axis_angle(rotations)
                        angles = angles[0]

                        #Save position raw
                        outName2 = f"{dirOut2}/sliced_{i}_{j.replace('.npz', '')}"
                        print(outName2)
                        pd.DataFrame(positions.reshape(-1, positions.shape[1]*positions.shape[2])).to_csv(f"{outName2}.csv", index = False, header = False)

                except Exception as error:
                        print(f"Error: {error}")

        print(f"Finished folder {k}")
