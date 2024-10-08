import sys

from data.phoneProcess.customFeatureExtract import get_second_derivative, extractFeats
#sys.path.insert(1, '/Users/pdealcan/Documents/github/EDGEk/')
sys.path.insert(1, './')

import numpy as np
import torch
import os
import pandas as pd
from pytorch3d.transforms import (axis_angle_to_matrix)
import json
import pickle as pkl
from dataset.dance_dataset import *
from vis import SMPLSkeleton, smplToPosition, smplTo6d, create_middle_marker, differentiate_fast, center_mean, translate
from pytorch3d.transforms import (axis_angle_to_quaternion, quaternion_apply,
                                  quaternion_multiply, quaternion_to_axis_angle, RotateAxisAngle)
from dataset.quaternion import ax_from_6v, quat_slerp

def slicer(df, rows, i):
    df_slice = {}
    df_slice['trans'] = df['trans'][i * rows : (i + 1) * rows, :]
    df_slice['poses'] = df['poses'][i * rows : (i + 1) * rows, :]
    df_slice['scaling'] = 1
    return df_slice

def slicer2(df, rows, i):
    df_slice = df[i * rows : (i + 1) * rows, :]
    return df_slice

#path = "/Users/pdealcan/Documents/github/data/CoE/accel/amass/amass_full/DanceDB/"
path = "./data/raw/amass/DanceDB/"
dirOutGround = "./data/test/motions_sliced/"
dirOutFeats = "./data/test/baseline_feats/"
datasets = os.listdir(path)

useAngles = True

if not os.path.exists(dirOutGround):
    os.makedirs(dirOutGround)
    print(f"Folder '{dirOutGround}' created.")

if not os.path.exists(dirOutFeats):
    os.makedirs(dirOutFeats)
    print(f"Folder '{dirOutFeats}' created.")

#Converting DanceDB to slices (full-body dance) and computing features from right thigh and
#left wrist
seconds = 10
newFreq = 30
if True:
    folders = os.listdir(f"{path}")
    for k in folders:
        if k != ".DS_Store":
            files = os.listdir(f"{path}/{k}")
            for j in files:
                if j != "shape.npz":
                    cFile = f"{path}/{k}/{j}"

                    substrings = ['Angry', 'Curiosity', 'Happy', 'Nervous', 'Sad', 'Scary', 'Excited', 'Annoyed', 'Bored', 'Miserable', 'Mix', 'Pleased', 'Relaxed', 'Sad', 'Satisfied', 'Tired', 'Afraid', 'Neutral']
                    should_filter = any(k in j for k in substrings) 
                    if(should_filter == False):
                        df = dict(np.load(cFile))
                        sr = df['mocap_framerate'] 

                        df['scaling'] = 1
                       
                        df['trans'] = df['trans']
                        df['poses'] = df['poses']

                        #Determine sampling rate stride
                        data_stride = int(sr //newFreq)

                        #Get angles only for output
                        if useAngles:

                            df['trans'] = center_mean(torch.from_numpy(df['trans']))
                            df['trans'] = df['trans'].numpy()

                            out_pos = smplTo6d(df['trans'], df['poses'], df['scaling'], aist = False)
                            out_pos = out_pos[:: data_stride, :] #Resampling to 30 fps

                        #Get position entire set
                        positions, rotations = smplToPosition(df['trans'], df['poses'], df['scaling'], aist = False)
                        positions = positions[0]
                        
                        #Resample
                        positions = positions[:: data_stride, :, :] #Resampling to 30 fps
                        sr = newFreq

                        #Getting acceleration from position
                        accel = positions.reshape(-1, positions.shape[1]*positions.shape[2])
                        #accel = differentiate_fast(accel, 2, newFreq)

                        phone = (accel[:,3:6] + accel[:,12:15])/2 #average of hip (marker 2) and knee (marker 5)
                        watch = accel[:,63:66] #position of wrist
                        IMUs = np.concatenate([phone, watch], 1)
                    
                        #Slicing
                        n_frames = IMUs.shape[0]
                        rows = int(sr*seconds)
                        num_chunks = int(n_frames / rows) #int() always rounds down, therefore we never get an unequal sample size
                        #Slicing
                        for i in range(1, num_chunks):
                           
                            #Save slice of position raw
                            outName = f"/sliced_{i}_{j.replace('.npz', '')}"
                            if(useAngles):
                                positions_sliced = slicer2(out_pos, rows, i)
                            else:
                                positions_sliced = slicer2(positions, rows, i)

                            #Getting IMUs
                            accel_sliced = slicer2(IMUs, rows, i)
                            original_shape = positions_sliced.shape

                            #Differentiating
                            accel_sliced = differentiate_fast(accel_sliced, 2, newFreq)

                            if useAngles:
                                print(f"Print output for AMASS is of shape: {positions_sliced.shape}")
                                pd.DataFrame(positions_sliced).to_pickle(f"{dirOutGround}/{outName}.pkl")
                            else:
                                positions_sliced = center_mean(positions_sliced).reshape(original_shape[0], original_shape[1], original_shape[2])
                                pd.DataFrame(positions_sliced.reshape(-1, positions_sliced.shape[1]*positions_sliced.shape[2])).to_pickle(f"{dirOutGround}/{outName}.pkl")

                            #Getting features
                            outName2 = f"/sliced_{i}_{j.replace('.npz', '')}"
                            df = extractFeats(accel_sliced, accel_sliced.shape[0])
                            print(df.shape)
                            df = np.float32(df)
                            np.save(f"{dirOutFeats}{outName2}", df)
                        else:
                            pass 
        print(f"Finished AMASS dataset")


## Process AIST++
#path = "/Users/pdealcan/Documents/github/data/CoE/accel/aistpp/motions/"
if True:
    path = "./data/raw/edge_aistpp/motions/"
    dirOutGround = "./data/train/motions_sliced/"
    dirOutFeats = "./data/train/baseline_feats/"
    filter_files = "./data/splits/ignore_list.txt"
    files = os.listdir(path)
    filter_files = pd.read_csv(filter_files)['files_ignore'].to_list()

    if not os.path.exists(dirOutGround):
        os.makedirs(dirOutGround)
        print(f"Folder '{dirOutGround}' created.")

    if not os.path.exists(dirOutFeats):
        os.makedirs(dirOutFeats)
        print(f"Folder '{dirOutFeats}' created.")

    for j in files:
        sr = 60
        newFreq = 30

        should_filter = any(j.replace('.pkl', '') in k for k in filter_files) 
        if(should_filter == False):
            df = pd.read_pickle(f"{path}{j}")

            #Resample stride
            data_stride = sr //newFreq

            #Convert to position
            if(useAngles):
                #Centering motion translations with zero mean
                df['smpl_trans'] = center_mean(torch.from_numpy(df['smpl_trans']))
                df['smpl_trans'] = df['smpl_trans'].numpy()

                out_pos = smplTo6d(df['smpl_trans'], df['smpl_poses'], df['smpl_scaling'], aist = True)
                out_pos = out_pos[:: data_stride, :] #Resampling to 30 fps

            positions, rotations = smplToPosition(df['smpl_trans'], df['smpl_poses'], df['smpl_scaling'], aist = True)
            positions = positions[0]

            #Resample positions
            positions = positions[:: data_stride, :, :] #Resampling to 30 fps
            sr = newFreq

            #Getting IMUs
            phone = [1, 4] #right_hip, right knee 
            watch = [21] #left wrist
            phone = create_middle_marker(positions, phone).reshape(-1, 3)
            watch = positions[:, watch, :].reshape(-1, 3)

            IMUs = torch.cat([phone, watch], dim = 1)

            #Get acceleration for thigh and wrist
            #accel = differentiate_fast(IMUs, 2, sr) #right thigh, left wrist

            #Slicing
            n_frames = IMUs.shape[0]
            rows = int(sr*seconds)
            num_chunks = int(n_frames / rows) #int() always rounds down, therefore we never get an unequal sample size
            for i in range(num_chunks):
                #Save slice of position raw
                outName = f"/sliced_{i}_{j.replace('.pkl', '')}"

                if useAngles:
                    positions_sliced = slicer2(out_pos, rows, i)
                else:
                    positions_sliced = slicer2(positions, rows, i)
                
                #Centering each segment in the mean
                original_shape = positions_sliced.shape

                if useAngles:
                    pd.DataFrame(positions_sliced).to_pickle(f"{dirOutGround}/{outName}.pkl")
                    print(f"Print output for AIST is of shape: {positions_sliced.shape}")
                else:
                    positions_sliced = center_mean(positions_sliced).reshape(original_shape[0], original_shape[1], original_shape[2])
                    pd.DataFrame(positions_sliced.reshape(-1, positions_sliced.shape[1]*positions_sliced.shape[2])).to_pickle(f"{dirOutGround}/{outName}.pkl")

                #Getting IMUs
                accel_sliced = slicer2(IMUs, rows, i)
                accel_sliced = differentiate_fast(accel_sliced, 2, sr) #right thigh, left wrist

                #Extract features
                outName2 = f"/sliced_{i}_{j.replace('.pkl', '')}"
                accel_sliced = extractFeats(accel_sliced, accel_sliced.shape[0])
                accel_sliced = np.float32(accel_sliced)
                np.save(f"{dirOutFeats}{outName2}", accel_sliced)
                #print("Saved " + outName2 + " features")
        else:
            print(j)



    #Acceleration computed
    #pd.DataFrame(input_data2.reshape(-1, 4*3)).to_csv("/Users/pdealcan/Documents/github/EDGEk/accelAndGyro2.csv", index = False, header = False)
