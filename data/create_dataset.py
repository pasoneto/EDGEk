import argparse

from accel_extraction_funcs import *

from filter_split_data import *
from slice import *

def create_dataset(type_feature, marker1 = None, marker2 = None, position_out = False):
    
    if type_feature == "position" and (marker1 == None or marker2 == None): 
        raise ValueError("Specify marker numbers")
    
    if position_out:
        print("Writting only positions")
    else:
        print("Writting angles and foot contact")

    slice_amass("../../../data/CoE/accel/amass/amass_full/DanceDB/", "../data/test/motions_sliced/", position_out=position_out)
    print("Finished test dataset")
    slice_aist("../../../EDGEk/data/raw/edge_aistpp/motions/", "../data/train/motions_sliced/", position_out = position_out)
    print("Finished train dataset")

    #process dataset to extract accel features
    print(f"Extracting {type_feature} features train")
    extract_features("../data/train/motions_sliced/", "../data/train/features/", type_feature, marker1 = marker1, marker2 = marker2, position_out = position_out, aist = True)
    print(f"Extracting {type_feature} features test")
    extract_features("../data/test/motions_sliced/", "../data/test/features/", type_feature, marker1 = marker1, marker2 = marker2, position_out = position_out, aist = False)

parser = argparse.ArgumentParser()
parser.add_argument("--type_feature", default="accelerometer")
parser.add_argument("--marker1", default=None)
parser.add_argument("--marker2", default=None)
parser.add_argument("--position_out", action="store_true", help="Set to True to enable position output")

opt = parser.parse_args()
create_dataset(type_feature = opt.type_feature, marker1 = opt.marker1, marker2 = opt.marker2, position_out = opt.position_out)
