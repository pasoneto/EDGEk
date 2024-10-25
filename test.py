import glob
import os
from functools import cmp_to_key
from pathlib import Path
from tempfile import TemporaryDirectory
import random

import jukemirlib
import numpy as np
import torch
from tqdm import tqdm

from args import parse_test_opt
from EDGE import EDGE
from data.audio_extraction.baseline_features import extract as baseline_extract
from data.audio_extraction.jukebox_features import extract as juke_extract

# sort filenames that look like songname_slice{number}.ext
key_func = lambda x: int(os.path.splitext(x)[0].split("_")[-1].split("slice")[-1])


def stringintcmp_(a, b):
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


stringintkey = cmp_to_key(stringintcmp_)

def test(opt):
    sample_length = opt.out_length
    sample_size = int(sample_length / 2.5) - 1

    temp_dir_list = []
    all_cond = []
    all_filenames = []
    print("Using precomputed features")
    # all subdirectories
    dir_list = ["/Users/pdealcan/Documents/github/edge_redo/EDGEk/data/test/features/"]
    for d in dir_list:
        file_list = sorted(glob.glob(f"{d}/*.pkl"), key=stringintkey)
        juke_file_list = sorted(glob.glob(f"{d}/*.pkl"), key=stringintkey)
        assert len(file_list) == len(juke_file_list)
        # random chunk after sanity check
        rand_idx = random.randint(0, len(file_list) - sample_size)
        file_list = file_list[rand_idx : rand_idx + sample_size]
        juke_file_list = juke_file_list[rand_idx : rand_idx + sample_size]
        cond_list = [np.load(x, allow_pickle=True) for x in juke_file_list]
        all_filenames.append(file_list)
        all_cond.append(torch.from_numpy(np.array(cond_list)))
    
    directory_weight = "/Users/pdealcan/Downloads/train-redo-5000.pt"
    model = EDGE(opt.feature_type, directory_weight)
    model.eval()

    # directory for optionally saving the dances for eval
    fk_out = "./generated_dances/"
    print("Generating dances")
    for i in range(len(all_cond)):
        data_tuple = None, all_cond[i], all_filenames[i]
        model.render_sample(
            data_tuple, "test", opt.render_dir, render_count=-1, fk_out=fk_out, render=True
        )
    print("Done")
    torch.cuda.empty_cache()
    for temp_dir in temp_dir_list:
        temp_dir.cleanup()


if __name__ == "__main__":
    opt = parse_test_opt()
    test(opt)
