import os
import pickle
import numpy as np
from tqdm import tqdm
import pandas as pd

from accel_extraction_funcs import center_mean, add_foot_contact, differentiate_fast

from vis import smplToPosition

def resample(q, pos, sr):
    newFreq = 15 #Determine sampling rate stride
    data_stride = int(sr //newFreq)

    #Resampling
    q = q[:: data_stride, :]
    pos = pos[:: data_stride, :] 
    
    return q, pos

def slice_motion(motion_file, out_dir, aist, position_out):
    file_name = os.path.splitext(os.path.basename(motion_file))[0]
    if aist == False:
        motion = dict(np.load(motion_file))
        pos, q, scale = motion["trans"], motion["poses"], 1 #Maybe the inverse?
        pos = center_mean(pos)
        sr = motion['mocap_framerate'] 
        assert sr == 120
        #Removing additional info that comes with amass (?)
        q = q.reshape((q.shape[0], -1, 3))
        q = q[:, 0:24, :]
        q = q.reshape((q.shape[0], q.shape[1]*q.shape[2]))
        q, pos = resample(q, pos, sr = sr)
    else:
        motion = dict(np.load(f"{motion_file}", allow_pickle=True))
        pos, q, scale = motion['smpl_trans'], motion['smpl_poses'], motion['smpl_scaling']
        pos /= scale #Normalize root position
        pos = center_mean(pos)
        #Resampling
        sr = 60
        q, pos = resample(q, pos, sr = sr)

    #FK and differentiation
    sr = 15
    out, _ = smplToPosition(q, pos, 1, aist = aist)
    out_pos = out[0]
    out_accel = out[0]
    out_accel = differentiate_fast(out_accel, order = 2, sr = sr)

    #Slicing
    seconds = 5
    n_frames = out_accel.shape[0]
    rows = int(sr*seconds)
    num_chunks = int(n_frames / rows) #int() always rounds down, therefore we never get an unequal sample size
    init_sample = 0 if aist else 1  #Amass initiates with T pose. Cut off first sample
    for i in range(init_sample, num_chunks):
        out_slice = out_accel[i * rows : (i + 1) * rows, :]
        out_pos_slice = out_pos[i * rows : (i + 1) * rows, :] #same slice, but only position, no accel
        out_slice = np.float32(out_slice)
        out_pos_slice = np.float32(out_pos_slice)

        if position_out:
            out_slice = out_slice.reshape(-1, 24*3)
        else:
            pass 

        pickle.dump(out_slice, open(f"{out_dir.replace('motions_sliced', 'motions_sliced_accel')}{file_name}_slice{i}.pkl", "wb"))
        pickle.dump(out_pos_slice, open(f"{out_dir}/{file_name}_slice{i}.pkl", "wb"))

def slice_amass(file_dir, out_dir, position_out):
    folders = os.listdir(f"{file_dir}")
    substrings = ["shape", 'Angry', 'Curiosity', 'Happy', 'Nervous', 'Sad', 'Scary', 'Excited', 'Annoyed', 'Bored', 'Miserable', 'Mix', 'Pleased', 'Relaxed', 'Satisfied', 'Tired', 'Afraid', 'Neutral']
    amass_files = [
        os.path.join(file_dir, folder, file)
        for folder in folders
        if os.path.isdir(os.path.join(file_dir, folder))
        for file in os.listdir(os.path.join(file_dir, folder))
        if (file.endswith(".npz") or file.endswith(".npz")) and not any(sub in file for sub in substrings)
    ]
    for file in tqdm(amass_files):
        slice_motion(file, out_dir, aist = False, position_out = position_out)

def slice_aist(file_dir, out_dir, position_out):
    ignore_list = pd.read_csv("../../../EDGEk/data/splits/ignore_list.txt")['files_ignore'].to_list()
    all_files = [os.path.join(file_dir, f) for f in os.listdir(file_dir) if f.endswith('.pkl')]
    aist_files = [f for f in all_files if not any(ignored in f for ignored in ignore_list)]
    for file in tqdm(aist_files):
        slice_motion(file, out_dir, aist = True, position_out = position_out)

