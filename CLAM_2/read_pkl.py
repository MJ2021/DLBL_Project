import pandas as pd
from glob import glob
from pathlib import Path
import numpy as np


df = pd.read_pickle('/home/Drivessd2tb/dlbl_data/Mohit/HEATMAP_RESULTS/heatmap_raw_results/HEATMAP_OUTPUT/ABC/007084CN__20240220_124651/007084CN__20240220_124651_mask.pkl')
new = pd.DataFrame.from_dict(df)
print(new.T)
    

