import numpy as np
import torch
from smplx import SMPLX
import trimesh
import vedo

import os
import pandas as pd

from pytorch3d.transforms import (axis_angle_to_matrix)
import json
import pickle as pkl
from dataset.dance_dataset import *
from vis import SMPLSkeleton
from pytorch3d.transforms import (axis_angle_to_quaternion, quaternion_apply,
                                  quaternion_multiply, quaternion_to_axis_angle, RotateAxisAngle)

from dataset.quaternion import ax_from_6v, quat_slerp
#from IMUPoser.src.imuposer.smpl.parametricModel import ParametricModel
#Adapted from: https://github.com/google/aistplusplus_api/blob/main/demos/run_vis.py

#Our rig has K = 23 joints, hence a pose
#is defined by 3 * 23 + 3 = 72 parameters; i.e. 3 for each part plus 
#3 for the root orientation. Let axis of rotation. 
#Then the axis angle for every joint j is transformed to a rotation matrix using 
#the Rodrigues formula

#Visualize mesh
def plotMESH(smpl_model, body_model, index, VERTEX_IDS):
#    faces = smplx_model.faces
    joints = body_model.joints.detach().numpy()[index]
    vertices = body_model.vertices.detach().numpy()[index] #Vertices are probably given as positions. Need quaternions
    mesha = trimesh.Trimesh(vertices)
#    l, w = vertices.shape
#    vertices = np.random.rand(l, w)
#    mesha.visual.face_colors = [200, 200, 250, 100]
    #Assign a color based on a scalar and a color map
    n = len(VERTEX_IDS)
    pc1 = vedo.Points(mesha.vertices[VERTEX_IDS], r=10)
    pc1.cmap("jet", list(range(n)))
    vedo.show((mesha, pc1), interactive=True)

def smplToPosition(pos, q, scale, aist = True):
    smpl = SMPLSkeleton()
    # to Tensor
    pos /= scale #Normalize by scale
    root_pos = torch.Tensor(np.array([pos]))
    local_q = torch.Tensor(np.array([q]))
    print(root_pos.shape)
    print(local_q.shape)
    # to ax
    bs, sq, c = local_q.shape
    local_q = local_q.reshape((bs, sq, -1, 3))
    # AISTPP dataset comes y-up - rotate to z-up to standardize against the pretrain dataset
    root_q = local_q[:, :, :1, :]  # sequence x 1 x 3 #Extracting the root axis angles
    root_q_quat = axis_angle_to_quaternion(root_q) #Converting to quaternions
    rotation = torch.Tensor(
        [0.7071068, 0.7071068, 0, 0]
    )  # 90 degrees about the x axis
    root_q_quat = quaternion_multiply(rotation, root_q_quat)
    root_q = quaternion_to_axis_angle(root_q_quat) #Back to quaternions
    local_q[:, :, :1, :] = root_q #Assign new rotated root
    # don't forget to rotate the root position too 😩
    pos_rotation = RotateAxisAngle(90, axis="X", degrees=True)
    root_pos = pos_rotation.transform_points(
        root_pos
    )  # basically (y, z) -> (-z, y), expressed as a rotation for readability
    # do FK
    # local_q: axis angle rotations for local rotation of each joint 
    # root_pos: root-joint positions
    positions = smpl.forward(local_q, root_pos)  # batch x sequence x 24 x 3
#    positions = positions[0]
#    positions = positions.numpy()
    return positions

#directoryIn = f"/Users/pdealcan/Documents/github/motions_sliced_original_train/"
directoryIn = f"/Users/pdealcan/Downloads/edge_aistpp/motions/"
files = os.listdir(directoryIn) #SMPL files
fName = files[45]
fName = "gJB_sBM_cAll_d07_mJB0_ch01.pkl"
df = pd.read_pickle(f"{directoryIn}{fName}")

df['smpl_trans'].shape #3
df['smpl_poses'].shape #72

positions, rotations = smplToPosition(df['smpl_trans'], df['smpl_poses'], df['smpl_scaling'])
rotations = torch.stack(rotations).permute(1, 2, 0, 3)

print(positions.shape)
print(rotations.shape)

angles = quaternion_to_axis_angle(rotations)
angles = angles[0]
angles = pd.DataFrame(angles.reshape(-1, 24*3))

pd.DataFrame(positions.reshape(-1, 24*3)).to_csv("/Users/pdealcan/Documents/github/EDGEk/testEDGEPositions.csv", index = False, header = False)
pd.DataFrame(rotations.reshape(-1, 24*4)).to_csv("/Users/pdealcan/Documents/github/EDGEk/testEDGERotations.csv", index = False, header = False)
pd.DataFrame(angles).to_csv("/Users/pdealcan/Documents/github/EDGEk/angles.csv", index = False, header = False)
