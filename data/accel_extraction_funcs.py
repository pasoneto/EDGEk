import sys

sys.path.insert(1, '/Users/pdealcan/Documents/github/edge_redo/EDGEk/')

import numpy as np
import torch
import os
import pandas as pd
from pytorch3d.transforms import (axis_angle_to_matrix)
import json
import pickle as pkl
from dataset.dance_dataset import *
from pytorch3d.transforms import (axis_angle_to_quaternion, quaternion_apply,
                                  quaternion_multiply, quaternion_to_axis_angle, RotateAxisAngle)

from vis import SMPLSkeleton, visu, smplToPosition
from dataset.quaternion import ax_from_6v, quat_slerp

from scipy.signal import butter, filtfilt

from sklearn.decomposition import PCA

from scipy.stats import skew, kurtosis
from scipy.fftpack import fft
from scipy.signal import find_peaks
import pywt  # PyWavelets library for wavelet transform

from tqdm import tqdm

def translate(df, offsets):
    df[:,0:df.shape[1]:3] = df[:,0:df.shape[1]:3] + offsets[0];
    df[:,1:df.shape[1]:3] = df[:,1:df.shape[1]:3] + offsets[1];
    df[:,2:df.shape[1]:3] = df[:,2:df.shape[1]:3] + offsets[2];
    return(df)

def center_mean(df):
    x = np.mean(np.mean(df[:,0:df.shape[1]:3], axis = 0))
    y = np.mean(np.mean(df[:,1:df.shape[1]:3], axis = 0))
    z = np.mean(np.mean(df[:,2:df.shape[1]:3], axis = 0))
    df = translate(df, [-x, -y, -z]);
    return(df)


def create_middle_marker(positions, indices):
    r"""
    Create a virtual marker between two other markers. 
        - positions: object holding marker positions (N_samples, n_markers, 3)
        - indices: an array of integers indicating two markers
    """
    markers = positions[:,indices,:]
    mid_point = np.mean([markers[:, 0, :], markers[:, 1, :]], axis=0)
    mid_point = torch.tensor(mid_point)
    return mid_point

#Getting IMUs
def extractIMUs(positions):
    phone = [1, 4] #right_hip, right knee 
    watch = [21] #left wrist
    phone = create_middle_marker(positions, phone).reshape(-1, 3)
    watch = positions[:, watch, :].reshape(-1, 3)
    IMUs = torch.cat([phone, watch], dim = 1)
    return IMUs

def extract2Markers(positions, marker1, marker2):
    marker1 = positions[:, marker1, :].reshape(-1, 3)
    marker2 = positions[:, marker2, :].reshape(-1, 3)
    markers = torch.cat([marker1, marker2], dim = 1)
    return markers

def differentiate_fast(d, order, sr):
    cutoff = .2;
    b, a = butter(2, cutoff, btype='lowpass', analog=False, output='ba')  # Butterworth filter coefficients
    for _ in range(order):
        d = np.diff(d, axis=0) #Difference between consecutive frames
        d = np.concatenate((np.tile(np.array([d[0]]), (1, 1)), d), axis=0) #Repeat first frame
        d = filtfilt(b, a, d, axis=0, padtype=None, padlen=0) #Butterworth filter
    d = d * (sr ** order)
    return d

def getPCAproj(x):
    pca = PCA(n_components=3)
    pca.fit(x)
    proj = pca.transform(x)
    return proj

# Peak Features
def fP(x):
    x, _ = find_peaks(x)
    return x

def extractFeats(acceleration_data, windowLength):
    #Descriptives accross euclidien dimensions (x, y, z)
    mean_values = np.mean(acceleration_data, axis=1)
    std_dev_values = np.std(acceleration_data, axis=1)
    skewness_values = skew(acceleration_data, axis=1)
    kurtosis_values = kurtosis(acceleration_data, axis=1)

    descriptivesColumns = np.column_stack([mean_values, std_dev_values, skewness_values, kurtosis_values])

    #Descriptives accross time
    mean_values = np.mean(acceleration_data, axis=0)
    std_dev_values = np.std(acceleration_data, axis=0)
    skewness_values = skew(acceleration_data, axis=0)
    kurtosis_values = kurtosis(acceleration_data, axis=0)

    descriptivesRows = np.concatenate([mean_values, std_dev_values, skewness_values, kurtosis_values])

    all0 = np.tile(descriptivesRows, (windowLength, 1))

    # Time-domain Features
    min_values = np.min(acceleration_data, axis=0)
    max_values = np.max(acceleration_data, axis=0)
    range_values = np.ptp(acceleration_data, axis=0)
    rms_values = np.sqrt(np.mean(acceleration_data**2, axis=0))
    energy_values = np.sum(acceleration_data**2, axis=0)
    variance_values = np.var(acceleration_data, axis=0)

    timeDescriptives = np.concatenate([min_values, max_values, range_values, rms_values, energy_values, variance_values])
    all1 = np.tile(timeDescriptives, (windowLength, 1))

    # Frequency-domain Features
    fft_values = np.abs(fft(acceleration_data, axis=0))

    # Time-Frequency Features (using Wavelet Transform)
    coeffs, _ = pywt.cwt(acceleration_data, np.arange(1, 10), 'gaus1')
    coeffs = np.column_stack(coeffs)

    frequencyDescriptives = np.column_stack([fft_values, coeffs])
    all2 = frequencyDescriptives

    # Signal Magnitude Area (SMA)
    sma_values = np.sum(np.abs(acceleration_data), axis=0)

    # Zero Crossing Rate
    zero_crossing_rate_values = np.sum(np.diff(np.sign(acceleration_data), axis=0) != 0, axis=0)

    magZero = np.concatenate([sma_values, zero_crossing_rate_values])
    all3 = np.tile(magZero, (windowLength, 1))

    # Correlation between Axes
    correlation_matrix = np.corrcoef(acceleration_data, rowvar=False)
    correlation_xy = correlation_matrix[0, 1]
    correlation_yz = correlation_matrix[1, 2]
    correlation_xz = correlation_matrix[0, 2]

    correlations = np.array([correlation_xy, correlation_yz, correlation_xz])
    correlations = np.tile(correlations, (windowLength, 1))

    peaks = [fP(acceleration_data[:, x]) for x in range(3)]  # Replace 0 with the axis of interest
    number_of_peaks = np.array([len(x) for x in peaks])

    nPeaks = np.tile(number_of_peaks, (windowLength, 1))
    nPeaks = np.column_stack([correlations, nPeaks])

    #PCA projections
    pcas = getPCAproj(acceleration_data)

    all4 = np.column_stack([nPeaks, pcas])
     
    allFeatures = np.column_stack([all0, all1, all2, all3, all4, acceleration_data])

    return allFeatures

def feat_extract(motion_file_sliced, output_feats, feature_type = "accelerometer", marker1 = None, marker2 = None, position_out = False, aist = True):
    file_name = os.path.splitext(os.path.basename(motion_file_sliced))[0]

    if position_out == False:
        motion = dict(np.load(motion_file_sliced, allow_pickle=True))
    else:
        motion = np.load(motion_file_sliced, allow_pickle=True)

    if position_out == False:
        pos, q, _ = motion["pos"], motion["q"], 1 #q: 3, pos: 24
        positions, _ = smplToPosition(q, pos, 1, aist = aist)
        positions = positions[0]
    else:
        positions = motion

    if feature_type == "accelerometer":
        IMUs = extractIMUs(positions)
        motion_sliced = differentiate_fast(IMUs, 2, sr = 30) #right thigh, left wrist
    elif feature_type == "positions":
        motion_sliced = extract2Markers(positions, marker1 = [marker1], marker2 = [marker2])
    else:
        raise Exception("Feature type not recognized")

    motion_sliced = extractFeats(motion_sliced, motion_sliced.shape[0])
    motion_sliced = np.float32(motion_sliced)
    pickle.dump(motion_sliced, open(f"{output_feats}/{file_name}.pkl", "wb"))

    return None

def extract_features(input_sliced, output_feats, feature_type = "accelerometer", marker1 = None, marker2 = None, position_out = False, aist = True):
    files = os.listdir(input_sliced)
    for file in tqdm(files):
        feat_extract(f"{input_sliced}{file}", output_feats, feature_type = feature_type, marker1 = marker1, marker2 = marker2, position_out = position_out, aist = aist)
    return None



