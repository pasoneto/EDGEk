import os
from pathlib import Path

from accel_extraction_funcs import *

from filter_split_data import *
from slice import *

def create_dataset():
    slice_amass("../../../data/CoE/accel/amass/amass_full/DanceDB/", "../data/test/motions_sliced/")
    print("Finished test dataset")
    slice_aist("../../../EDGEk/data/raw/edge_aistpp/motions/", "../data/train/motions_sliced/")
    print("Finished train dataset")

    #process dataset to extract accel features
    print("Extracting accelerometer features train")
    extract_features("../data/train/motions_sliced/", "../data/train/features/")

    print("Extracting accelerometer features test")
    extract_features("../data/test/motions_sliced/", "../data/test/features/")

#parser = argparse.ArgumentParser()
#parser.add_argument("--stride", type=float, default=0.5)
#parser.add_argument("--length", type=float, default=5.0, help="checkpoint")
#parser.add_argument(
#    "--dataset_folder",
#    type=str,
#    default="edge_aistpp",
#    help="folder containing motions and music",
#)
#parser.add_argument("--extract-baseline", action="store_true")
#parser.add_argument("--extract-jukebox", action="store_true")
#parser.add_argument("--extract-accel", action="store_true")
#opt = parser.parse_args()
#return opt

create_dataset()
