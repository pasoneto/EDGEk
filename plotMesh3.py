import numpy as np
import torch
import os
import sys
import pandas as pd
sys.path.append("../smplx/")
from smplx import SMPLX
#from smplx import SMPL
import trimesh
import vedo

import matplotlib.pyplot as plt
import matplotlib.cm as cm

import os
import pandas as pd
import json

# Import SMPLX class
from smplx import SMPLX

# Opening JSON file
f = open('./joint_ids.json')
joints = json.load(f)

joint_statistics = pd.read_csv("./eval/mpe_statistics_diff.csv")
print(joint_statistics)
def generate_colors(n):
    import random
    colors = []
    for _ in range(n):
        color = [random.randint(0, 255) for _ in range(3)] + [255]  # Ensure full opacity
        colors.append(color)
    return colors

def generate_color_list(values):
    cmap = cm.get_cmap('coolwarm')  # Choose a colormap (e.g., viridis, jet, etc.)
    norm = plt.Normalize(min(values), max(values))  # Normalize your values
    scalar_map = cm.ScalarMappable(norm=norm, cmap=cmap)
    colors = [list(scalar_map.to_rgba(val)) for val in values]
    a_colors = []
    for i in colors:
        new_list = [int(x * 255) for x in i]
        a_colors.append(new_list)
    return a_colors

def plotMESH(smpl_model, body_model, index, joints, joint_statistics):
    vertices = body_model.vertices.detach().numpy()[index]
    faces = smpl_model.faces
    
    # Create the mesh
    mesha = trimesh.Trimesh(vertices, faces)

    # Function to find faces that contain the given vertex IDs
    def get_faces_from_vertex_ids(vertex_ids):
        face_mask = np.isin(faces, vertex_ids).any(axis=1)
        return np.where(face_mask)

    #face_names = ['head', 'leftHand', 'rightHand', 'leftHandIndex1', 'rightHandIndex1']

    face_names = joint_statistics['face_name']
    faces_list = [get_faces_from_vertex_ids(joints[x]) for x in face_names]

    values = joint_statistics['sd'].to_numpy().tolist()
    colors = generate_color_list(values)

    face_colors = np.ones((mesha.faces.shape[0], 4)) * 255  # Default color (white)
    
    # Create a mask for the face colors
    for l in range(len(faces_list)):
        print(face_names[l])
        face_colors[faces_list[l]] = colors[l] 

    #Change here for different statistics
    face_colors[get_faces_from_vertex_ids(joints['rightHandIndex1'])] = colors[16]#[179, 3, 38, 255]
    face_colors[get_faces_from_vertex_ids(joints['leftHandIndex1'])] = colors[6]#[58, 76, 192, 255]

    mesha.visual.face_colors = face_colors

    # Visualize the mesh
    vedo.show(mesha, interactive=True)
    

directoryIn = "/Users/pdealcan/Documents/github/data/CoE/accel/aistpp/motions_raw/"
files = os.listdir(directoryIn)  # SMPL files
data = pd.read_pickle(f"{directoryIn}{files[20]}")

trans, poses, scaling = data['smpl_trans'], data['smpl_poses'], data['smpl_scaling']
n_frames = poses.shape[0]
poses = poses.reshape(-1, 24, 3)

poses = torch.as_tensor(poses, dtype=torch.float32)
trans = torch.as_tensor(trans, dtype=torch.float32)

smplx_model = SMPLX('../amass/support_data/smplx/SMPLX_MALE.npz',
                    flat_hand_mean=True,
                    use_pca=False,
                    batch_size=n_frames).eval()

bodyModel = smplx_model.forward(
        global_orient=poses[:,0:1],
        body_pose=poses[:,1:22],
        transl=trans,
        scaling=scaling.reshape(1, 1),
        )

# Define vertex IDs for heads and arms. These IDs are examples and need to be adjusted to match the actual vertex IDs.
index = 55
plotMESH(smplx_model, bodyModel, index, joints, joint_statistics)
