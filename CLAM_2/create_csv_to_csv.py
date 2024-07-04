import pandas as pd
from glob import glob
from pathlib import Path
import numpy as np
import h5py

csv_file_path = '/home/Drivessd2tb/dlbl_data/Mohit/FEATURES_FINAL/csv_files_with_white_filter/003128CN__20240214_131534.csv'
save_dir = '/home/Drivessd2tb/dlbl_data/Mohit/FEATURES_FINAL/csv_files_with_white_filter_for_json/'

data_record = {'x' : [], 'y' : [], 'attention' : [], 'bin_VAL' : [], 'CLR' : []}

df = pd.read_csv(csv_file_path)
print(len(df))
for i in range(len(df)):
    data_record['x'].append(df['x'][i])
    data_record['y'].append(df['y'][i])
    data_record['attention'].append(1)
    data_record['bin_VAL'].append(1)
    data_record['CLR'].append(1)
# data_record['x'], data_record['y'] = df['x'], df['y']
# data_record['attention'] = {1}*2130
# data_record['bin_val'] = {1}*2130
# data_record['CLR'] = {1}*2130

df2 = pd.DataFrame(data_record)
slide = csv_file_path.split('.')[0]
slide_name = slide.split('/')[7]
df2.to_csv(save_dir + slide_name + '.csv', sep = ',', index = False)
