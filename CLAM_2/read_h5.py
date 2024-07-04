import pandas as pd
from glob import glob
from pathlib import Path
import numpy as np
import h5py

#path= '/home/Drivessd2tb/dlbl_data/Mohit/FEATURES_FINAL/h5_files/003611CN__20240214_125033.h5'
# path1='/workspace/hpv_project/feature_tmh_w1/tumor_vs_normal_resnet_features/h5_files/CAIB-T00001402OP01B01P0101HE.h5'
# path2='/workspace/hpv_project/feature_tmh_w1/tumor_vs_normal_resnet_features/h5_files/CAIB-T00001419OP01B03P0101HE.h5'

h5_files_path = glob('/home/Drivessd2tb/Mohit2_new/Created_Patches_new/patches/*', recursive = True)
# h5_files_path = glob('/home/Drivessd2tb/dlbl_data/Mohit/FEATURES_FINAL/h5_files_with_white_filter/*', recursive = True)
# for i in range(len(h5_files_path)):
for i in range(len(h5_files_path)):
    h5_path = h5_files_path[i]
    print("H5_PATH IS", h5_path)
    slide = h5_path.split('.')[0]
    print("SLIDE IS", slide)
    with h5py.File(h5_path, 'r') as f:
        data = f['coords']
        print(len(data))
        # print(f.keys())
        data_record = {'x' : [],'y' : []}
        for i in range(len(data)):
            data1= data[i]
            data1=data1.tolist()
            # print(data1)
            # data_record['coords'].append(data1)
            data_record['x'].append(data1[0])
            data_record['y'].append(data1[1])
    df = pd.DataFrame(data_record)
    slide_name = slide.split('/')[6]
    print("SLIDE NAME IS", slide_name)
    sv_file = '/home/Drivessd2tb/Mohit2_new/csv_files_without_white_filter_from_patches_new/' + slide_name +'.csv'
    df.to_csv(sv_file, sep = ',', index=False)

