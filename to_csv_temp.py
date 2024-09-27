import pandas as pd
import os

f = os.listdir("./generatedDance/sliced_predicted/")
file_names = [x.replace("csv", "pkl") for x in f]
files = [pd.read_pickle("./data/test/motions_sliced/" + a) for a in file_names]
for k in range(len(f)):
    files[k].to_csv(f"./data/original_csv/{f[k]}", index = False, header = False)

