import numpy as np
import torch
from smplx import SMPLX
#from smplx import SMPL
import trimesh
import vedo

import os
import pandas as pd

from pytorch3d.transforms import (axis_angle_to_matrix)

#from IMUPoser.src.imuposer.smpl.parametricModel import ParametricModel
#Adapted from: https://github.com/google/aistplusplus_api/blob/main/demos/run_vis.py

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


directoryIn = "/Users/pdealcan/Documents/github/data/CoE/accel/aistpp/"
files = os.listdir(directoryIn) #SMPL files
data = pd.read_pickle(f"{directoryIn}{files[0]}")

data['smpl_trans'].shape #3
data['smpl_poses'].shape #72

trans, poses, scaling = data['smpl_trans'], data['smpl_poses'], data['smpl_scaling']
n_frames = poses.shape[0]
poses = poses.reshape(-1, 24, 3)

#betas = torch.as_tensor(betas, dtype=torch.float32)
poses = torch.as_tensor(poses, dtype=torch.float32)
trans = torch.as_tensor(trans, dtype=torch.float32)

smplx_model = SMPLX('../amass/support_data/smplx/SMPLX_MALE.npz',
#                    num_betas=betas.shape[0],
                    flat_hand_mean=True,
                    use_pca=False,
                    batch_size=n_frames).eval()

#Pose: axis-angle format
#Getting only 21 joints. Verify this
print(poses.shape)
bodyModel = smplx_model.forward(
        global_orient=poses[:,0:1],
        body_pose=poses[:,1:22],
        transl=trans,
        scaling=scaling.reshape(1, 1),
        )

pd.DataFrame(bodyModel.joints.detach().numpy().reshape(-1, 127*3)).to_csv("/Users/pdealcan/Documents/github/EDGEk/test.csv", index = False, header = False)

#Wrist and right leg. AIST++
VERTEX_IDS = [4583, 6265]
index = 100
plotMESH(smplx_model, bodyModel, index, VERTEX_IDS)


#Both vertices and joints are represented in angle rotation

#p = axis_angle_to_matrix(poses.reshape(-1, 24, 3)) #Following AIST++
#p.shape
#smplx_model.forward(p)

#VERTEX_IDS = [4583, 6265]
#vertices_selected = bodyModel.vertices[:,VERTEX_IDS,:]
#_syn_acc(vertices_selected) #Calculate acceleration

#grot, joint, vert = body_model.forward_kinematics(p, shape[i], tran[b:b + l], calc_mesh=True)
smplx_model.forward(p, 0, tran, calc_mesh=True)
out_pose.append(pose[b:b + l].clone())  # N, 24, 3
out_tran.append(tran[b:b + l].clone())  # N, 3
out_shape.append(shape[i].clone())  # 10
out_joint.append(joint[:, :24].contiguous().clone())  # N, 24, 3
out_vacc.append(_syn_acc(vert[:, vi_mask]))  # N, 6, 3
out_vrot.append(grot[:, ji_mask])  # N, 6, 3, 3















# left wrist, right thigh
#vi_mask = torch.tensor([1961, 4362])
#ji_mask = torch.tensor([18, 2])

framerate = int(60)

#tran = torch.tensor(np.asarray(trans, np.float32))
#pose = torch.tensor(np.asarray(poses, np.float32)).view(-1, 24, 3)

#Stopped here
#print('Synthesizing IMU accelerations and orientations')
#out_pose, out_shape, out_tran, out_joint, out_vrot, out_vacc = [], [], [], [], [], []

#Check if this is the same as 6dof from AIST++
#p = axis_angle_to_rotation_matrix(pose).view(-1, 24, 3, 3)





























#video = vedo.Video("./spider.mp4", duration=4, backend='ffmpeg') # backend='opencv'

#vedo.show(pc1, interactive=True)
