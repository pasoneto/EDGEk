#Computes global rotations of joints using the method provided in EDGE 
import numpy as np
import torch
import trimesh
import vedo
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

import numpy as np
from scipy.signal import butter, filtfilt

directoryIn = "/Users/pdealcan/Documents/github/data/CoE/accel/aistpp/motions_raw/"
files = os.listdir(directoryIn) #SMPL files
df = pd.read_pickle(f"{directoryIn}{files[0]}")
sr = 30
positions, rotations = smplToPosition(df['smpl_trans'], df['smpl_poses'], df['smpl_scaling'], aist = True)
positions = positions[0]

angles = quaternion_to_axis_angle(rotations)
angles = angles[0]

##Getting joint orientation
joint_interest_one = [1, 21] #right_hip, left wrist (phone one side, watch other)
joint_interest_two = [2, 20] #left_hip, right wrist(phone one side, watch other)

joint_rotations_one = angles[:,joint_interest_one,:]
joint_rotations_one = differentiate_fast(joint_rotations_one, 1, sr)

joint_rotations_two = angles[:,joint_interest_two,:]
joint_rotations_two = differentiate_fast(joint_rotations_two, 1, sr)

#Extracting acceleration from midpoint between knee and hip
right_thigh = [1, 4]
left_thigh = [2, 5]

right_thigh = create_middle_marker(positions, right_thigh).reshape(-1, 1, 3)
left_thigh = create_middle_marker(positions, left_thigh).reshape(-1, 1, 3)

positions = torch.cat([positions, right_thigh, left_thigh], dim = 1)

#Get acceleration for thigh and wrist
acceleration_one = differentiate_fast(positions[:,[24, 20],:], 2, sr) #right thigh, left wrist
acceleration_two = differentiate_fast(positions[:,[25, 21],:], 2, sr) #left thigh, right wrist

set1 = [torch.tensor(acceleration_one), torch.tensor(joint_rotations_one)] #phone on the right, watch left
set2 = [torch.tensor(acceleration_two), torch.tensor(joint_rotations_two)] #phone on the left, watch right

#rThighACC, lWristACC, rThighRot, lWristRot
#1, 2, 3, 4
input_data1 = torch.cat(set1, dim=1)
input_data2 = torch.cat(set2, dim=1)

pd.DataFrame(positions.reshape(-1, positions.shape[1]*positions.shape[2])).to_csv("/Users/pdealcan/Documents/github/EDGEk/pos.csv", index = False, header = False)

#Acceleration computed
pd.DataFrame(input_data2.reshape(-1, 4*3)).to_csv("/Users/pdealcan/Documents/github/EDGEk/accelAndGyro2.csv", index = False, header = False)

