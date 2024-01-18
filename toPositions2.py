import pandas as pd
import os
import numpy as np
import torch
import sys
import os

import json

import pickle as pkl

from dataset.dance_dataset import *

from vis import SMPLSkeleton

from pytorch3d.transforms import (axis_angle_to_quaternion, quaternion_apply,
                                  quaternion_multiply, quaternion_to_axis_angle, RotateAxisAngle)

from dataset.quaternion import ax_to_6v, quat_from_6v
# CONVERTS SMPL REPRESENTATIONS TO POSITIONS, AND CREATES A ROOT FOR THE CENTER OF
# A PHONE AS THE AVERAGE OF HIP AND KNEE MARKERS

#markers = ["root", "lhip", "rhip", "belly", "lknee", "rknee", "spine", "lankle", "rankle", "chest", "ltoes", "rtoes", "neck", "linshoulder", "rinshoulder", "head",  "lshoulder", "rshoulder", "lelbow", "relbow", "lwrist", "rwrist", "lhand", "rhand"]
def smplToPosition(root_pos, local_q, position = True):
    # FK skeleton
    smpl = SMPLSkeleton()
    # to Tensor
    root_pos = torch.Tensor(root_pos)
    local_q = torch.Tensor(local_q)
    # to ax
    bs = 1
    sq, c = local_q.shape
    local_q = local_q.reshape((bs, sq, -1, 3))
    # AISTPP dataset comes y-up - rotate to z-up to standardize against the pretrain dataset
    root_q = local_q[:, :, :1, :]  # sequence x 1 x 3
    root_q_quat = axis_angle_to_quaternion(root_q)
    rotation = torch.Tensor(
        [0.7071068, 0.7071068, 0, 0]
    )  # 90 degrees about the x axis
    root_q_quat = quaternion_multiply(rotation, root_q_quat)
    root_q = quaternion_to_axis_angle(root_q_quat)
    local_q[:, :, :1, :] = root_q
    # don't forget to rotate the root position too 😩
    pos_rotation = RotateAxisAngle(90, axis="X", degrees=True)
    root_pos = pos_rotation.transform_points(
        root_pos
    )  # basically (y, z) -> (-z, y), expressed as a rotation for readability
    if position:
        root_pos = torch.tensor(np.array([root_pos]))
        positions = smpl.forward(local_q, root_pos)  # batch x sequence x 24 x 3
        return positions[0]
    else:
        # to 6d
        local_q = ax_to_6v(local_q)
        local_q = quat_from_6v(local_q)[0]
    #    root_pos = axis_angle_to_quaternion(root_q) #My test.
        return local_q

#train = "test"
directoryIn = f"./data/accel/test/motions/"
files = os.listdir(directoryIn) #SMPL files
df = pd.read_pickle(f"{directoryIn}{files[0]}")

#Pos: 3
#q: 72

#Get quaternions
a = smplToPosition(df['pos'], df['q'], False)
quaternions = a.reshape(720, -1)
quaternions = pd.DataFrame(quaternions)
quaternions.shape

#Get positions
b = smplToPosition(df['pos'], df['q'], True)
positions = b.reshape(720, -1)
positions = pd.DataFrame(quaternions)
positions.shape

positions.to_csv("/Users/pdealcan/Documents/github/EDGEk/positions.csv", index = False, header = False)
quaternions.to_csv("/Users/pdealcan/Documents/github/EDGEk/quaternion.csv", index = False, header = False)
