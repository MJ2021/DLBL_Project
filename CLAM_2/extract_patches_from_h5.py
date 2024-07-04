import pandas as pd
from glob import glob
from pathlib import Path
import numpy as np
import h5py

path= '/home/Drivessd2tb/Mohit2_new/HEATMAP_RESULTS/heatmap_raw_results/HEATMAP_OUTPUT/GCB/011487CZ__20240628_091310/011487CZ__20240628_091310_blockmap.h5'
# path1='/workspace/hpv_project/feature_tmh_w1/tumor_vs_normal_resnet_features/h5_files/CAIB-T00001402OP01B01P0101HE.h5'
# path2='/workspace/hpv_project/feature_tmh_w1/tumor_vs_normal_resnet_features/h5_files/CAIB-T00001419OP01B03P0101HE.h5'

# h5_files_path = glob('/home/Drivessd2tb/Mohit2_new/Created_Patches_new/patches/*', recursive = True)
# h5_files_path = glob('/home/Drivessd2tb/dlbl_data/Mohit/FEATURES_FINAL/h5_files_with_white_filter/*', recursive = True)

# for i in range(len(h5_files_path)):
#     h5_path = h5_files_path[i]
h5_path = path
print("H5_PATH IS", h5_path)
slide = h5_path.split('.')[0]
print("SLIDE IS", slide)
with h5py.File(h5_path, 'r') as f:
    data = f['coords']
    attention = f['attention_scores']
    print(attention)
    print(len(data))
    # print(f.keys())
    data_record = {'x' : [],'y' : [], 'attention' : []}
    for i in range(len(data)):
        data1= data[i]
        data1=data1.tolist()
        # print(data1)
        # data_record['coords'].append(data1)
        data_record['x'].append(data1[0])
        data_record['y'].append(data1[1])
        data_record['attention'].append(attention[i][0])

df = pd.DataFrame(data_record)
slide_name = slide.split('/')[8]
class_name = slide.split('/')[7]
print("SLIDE NAME IS", slide_name)
sv_file = '/home/Drivessd2tb/Mohit_Combined/csv_files_from_heatmap_h5/' + class_name + '/' + slide_name +'.csv'
df.to_csv(sv_file, sep = ',', index=False)

