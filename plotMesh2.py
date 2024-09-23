import numpy as np
import torch
import os
import sys
sys.path.append("../smplx/")
from smplx import SMPLX
#from smplx import SMPL
import trimesh
import vedo

import os
import pandas as pd

from pytorch3d.transforms import (axis_angle_to_matrix)

# Import SMPLX class
from smplx import SMPLX

def find_faces_by_vertices(mesh, vertex_ids):
    face_ids = []
    for face_id, face in enumerate(mesh.faces):
        if any(vertex in face for vertex in vertex_ids):
            face_ids.append(face_id)
    return face_ids

def plotMESH(smpl_model, body_model, index):
    vertices = body_model.vertices.detach().numpy()[index]
    faces = smpl_model.faces
    
    # Create the mesh
    mesha = trimesh.Trimesh(vertices, faces)
    
    # Create a mask for the face colors
    face_colors = np.ones((mesha.faces.shape[0], 4)) * 255  # Default color (white)
    
    # Assign colors to head, left arm, and right arm

    #face_colors[range(6, 9120)] = [0, 0, 255, 255]

    #face_colors[range(10000, 20000)] = [0, 0, 255, 255]
    #face_colors[head_faces] = [255, 0, 0, 255]  # Red color for the head
    #face_colors[left_arm_faces] = [0, 0, 255, 255]  # Blue color for the left arm
    #face_colors[right_arm_faces] = [0, 255, 0, 255]  # Green color for the right arm

    mesha.visual.face_colors = face_colors

    # Visualize the mesh
    vedo.show(mesha, interactive=True)

directoryIn = "/Users/pdealcan/Documents/github/data/CoE/accel/aistpp/motions_raw/"
files = os.listdir(directoryIn)  # SMPL files
data = pd.read_pickle(f"{directoryIn}{files[0]}")

trans, poses, scaling = data['smpl_trans'], data['smpl_poses'], data['smpl_scaling']
n_frames = poses.shape[0]
poses = poses.reshape(-1, 24, 3)

poses = torch.as_tensor(poses, dtype=torch.float32)
trans = torch.as_tensor(trans, dtype=torch.float32)

smplx_model = SMPLX('../amass/support_data/smplx/SMPLX_MALE.npz',
                    flat_hand_mean=True,
                    use_pca=True,
                    use_face_contour=False,
                    batch_size=n_frames).eval()


bodyModel = smplx_model.forward(
        global_orient=poses[:,0:1],
        body_pose=poses[:,1:22],
        transl=trans,
        scaling=scaling.reshape(1, 1),
        )


2319
# Define face IDs for heads and arms. These IDs are examples and need to be adjusted to match the actual face IDs.

index = 0

plotMESH(smplx_model, bodyModel, index)
