import os
import numpy as np

base = os.listdir("./predicted_amass/")
base = np.unique(["_".join(k.split('_')[2:]) for k in base])
str_out = ['sliced_0_', 'Angry', 'Curiosity', 'Happy', 'Nervous', 'Sad', 'Scary', 'Excited', 'Annoyed', 'Bored', 'Miserable', 'Mix', 'Pleased', 'Relaxed', 'Sad', 'Satisfied', 'Tired', 'Afraid', 'Neutral']
base = [s for s in base if not any(sub in s for sub in str_out)]
files_raw = os.listdir("./predicted_amass/")
files_raw = [s for s in files_raw if not any(sub in s for sub in str_out)]
len(files_raw)
final_selection = []
for k in base:
    base_sel = [n for n in files_raw if k in n]
    random_element = np.random.randint(0, len(base_sel))
    final_selection.append(base_sel[random_element])

import pandas as pd
pd.DataFrame({"selected_dances": final_selection}).to_csv("./random_selection_dances_perceptual_experiment.csv")

#For the perceptual experiment, we selected a random sample of 43 clips from the test dataset
#This was done by selecting a single 10 seconds clip from each of the 43 performances
#that composed our selection of DanceDB dataset. Again, we excluded the initial 10 seconds segment of each dance.

