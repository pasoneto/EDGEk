import glob
import os
import pickle
import random
from functools import cmp_to_key
from pathlib import Path
from typing import Any

import numpy as np
import torch
from pytorch3d.transforms import (RotateAxisAngle, axis_angle_to_quaternion,
                                  quaternion_multiply,
                                  quaternion_to_axis_angle)
from torch.utils.data import Dataset

from dataset.preprocess import Normalizer, vectorize_many
from dataset.quaternion import ax_to_6v
from vis import SMPLSkeleton


class AISTPPDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        backup_path: str,
        train: bool,
        feature_type: str = "jukebox",
        normalizer: Any = None,
        data_len: int = -1,
        include_contacts: bool = True,
        force_reload: bool = False,
    ):
        self.data_path = data_path
        self.raw_fps = 60
        self.data_fps = 30
        assert self.data_fps <= self.raw_fps
        self.data_stride = self.raw_fps // self.data_fps

        self.train = train
        self.name = "Train" if self.train else "Test"
        self.feature_type = feature_type

        self.normalizer = normalizer
        self.data_len = data_len

        backup_path = Path(backup_path)
        backup_path.mkdir(parents=True, exist_ok=True)
        # save normalizer
        if not train:
            pickle.dump(
                normalizer, open(os.path.join(backup_path, "normalizer.pkl"), "wb")
            )

        print("Loading dataset...")
        data = self.load_aistpp()  # Call this last

        print(
            f"Loaded {self.name} Dataset With Dimensions: Pos: {data['pos'].shape}"
        )

        # process data, convert to 6dof etc
        pose_input = self.process_dataset(torch.from_numpy(data["pos"]))
        
        self.data = {
            "pose": pose_input,
            "filenames": data["filenames"],
        }
        assert len(pose_input) == len(data["filenames"])
        self.length = len(pose_input)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        filename_ = self.data["filenames"][idx]
        feature = torch.from_numpy(np.load(filename_, allow_pickle=True))
        return self.data["pose"][idx], feature, filename_

    def load_aistpp(self):
        # open data path
        split_data_path = os.path.join(
            self.data_path, "train" if self.train else "test"
        )

        # Structure:
        # data
        #   |- train
        #   |    |- motion_sliced
        #   |    |- wav_sliced
        #   |    |- baseline_features
        #   |    |- jukebox_features
        #   |    |- motions
        #   |    |- wavs

        motion_path = os.path.join(split_data_path, "motions_sliced")
        features = os.path.join(split_data_path, f"features")

        # sort motions and sounds
        motions = sorted(glob.glob(os.path.join(motion_path, "*.pkl")))
        features = sorted(glob.glob(os.path.join(features, "*.pkl")))

        # stack the motions and features together
        all_pos = []
        all_names = []
        assert len(motions) == len(features)
        for motion, feature in zip(motions, features):
            # make sure name is matching
            m_name = os.path.splitext(os.path.basename(motion))[0]
            f_name = os.path.splitext(os.path.basename(feature))[0]
            assert m_name == f_name, str((motion, feature))
            # load motion
            data = pickle.load(open(motion, "rb"))
            #data = data.reshape(data.shape[0], data.shape[1]*data.shape[2]) 
            #If position is out, i need to reshape
            all_pos.append(data)
            all_names.append(feature)

        all_pos = np.array(all_pos)  # N x seq x 3
        data = {"pos": all_pos, "filenames": all_names}
        return data

    def process_dataset(self, data):

        global_pose_vec_input = data
        print(f"Received processed data with shape {global_pose_vec_input.shape}")

        # normalize the data. Both train and test need the same normalizer.
        if self.train:
            self.normalizer = Normalizer(global_pose_vec_input)
        else:
            assert self.normalizer is not None
        global_pose_vec_input = self.normalizer.normalize(global_pose_vec_input)

        assert not torch.isnan(global_pose_vec_input).any()
        data_name = "Train" if self.train else "Test"

        global_pose_vec_input = global_pose_vec_input

        print(f"{data_name} Dataset Motion Features Dim: {global_pose_vec_input.shape}")

        return global_pose_vec_input


class OrderedMusicDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        train: bool = False,
        feature_type: str = "baseline",
        data_name: str = "aist",
    ):
        self.data_path = data_path
        self.data_fps = 30
        self.feature_type = feature_type
        self.test_list = set(
            [
                "mLH4",
                "mKR2",
                "mBR0",
                "mLO2",
                "mJB5",
                "mWA0",
                "mJS3",
                "mMH3",
                "mHO5",
                "mPO1",
            ]
        )
        self.train = train

        # if not aist, then set train to true to ignore test split logic
        self.data_name = data_name
        if self.data_name != "aist":
            self.train = True

        self.data = self.load_music()  # Call this last

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return None

    def get_batch(self, batch_size, idx=None):
        key = random.choice(self.keys) if idx is None else self.keys[idx]
        seq = self.data[key]
        if len(seq) <= batch_size:
            seq_slice = seq
        else:
            max_start = len(seq) - batch_size
            start = random.randint(0, max_start)
            seq_slice = seq[start : start + batch_size]

        # now we have a batch of filenames
        filenames = [os.path.join(self.music_path, x + ".pkl") for x in seq_slice]
        # get the features
        features = np.array([np.load(x) for x in filenames])

        return torch.Tensor(features), seq_slice

    def load_music(self):
        # open data path
        split_data_path = os.path.join(self.data_path)
        music_path = os.path.join(
            split_data_path,
            f"{self.data_name}_baseline_feats"
            if self.feature_type == "baseline"
            else f"{self.data_name}_juke_feats/juke_66",
        )
        self.music_path = music_path
        # get the music filenames strided, with each subsequent item 5 slices (2.5 seconds) apart
        all_names = []

        key_func = lambda x: int(x.split("_")[-1].split("e")[-1])

        def stringintcmp(a, b):
            aa, bb = "".join(a.split("_")[:-1]), "".join(b.split("_")[:-1])
            ka, kb = key_func(a), key_func(b)
            if aa < bb:
                return -1
            if aa > bb:
                return 1
            if ka < kb:
                return -1
            if ka > kb:
                return 1
            return 0

        for features in glob.glob(os.path.join(music_path, "*.pkl")):
            fname = os.path.splitext(os.path.basename(features))[0]
            all_names.append(fname)
        all_names = sorted(all_names, key=cmp_to_key(stringintcmp))
        data_dict = {}
        for name in all_names:
            k = "".join(name.split("_")[:-1])
            if (self.train and k in self.test_list) or (
                (not self.train) and k not in self.test_list
            ):
                continue
            data_dict[k] = data_dict.get(k, []) + [name]
        self.keys = sorted(list(data_dict.keys()))
        return data_dict
