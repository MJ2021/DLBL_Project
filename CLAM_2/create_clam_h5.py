import numpy as np
import sys
import argparse
from glob import glob
import torch
import torch.nn as nn
import os
import pandas as pd
# from openslide import open_slide
import h5py
from tqdm import tqdm
# from torchvision import summary

ps=256
ds=1

#csvs = glob('/workspace/clam1/CLAM/csv_tmc_w_filt_pma_annot/*.csv')
csvs = glob('/home/Drivessd2tb/Mohit_Combined/csv_files_with_white_filter_from_patches_combined_210/*', recursive = True)
save_dir = '/home/Drivessd2tb/Mohit_Combined/h5_files_with_white_filter_from_patches_combined_210/patches/'
print(len(csvs))
#sys.exit()
for i in range(len(csvs)):
    # exit()
    df = pd.read_csv(csvs[i])
    x, y = df['x'], df['y']
    coords = np.zeros((len(x),2), int)
    coords[:,0] = np.array(x)
    coords[:,1] = np.array(y)
    slide = csvs[i].split('.')[0]
    slide_name = slide.split('/')[5]
    print("SLIDE NAME IS ", slide_name)
    h5_save_path = save_dir + slide_name +'.h5'
    
    ps=256
    ds=1
    # s_name=path_csv_class.split('/')[-1][:-4]
    attr = {'patch_size' :            ps, # To be considered...
        'patch_level' :           0,
        'downsample':             ds,
        # 'downsampled_level_dim' : (155,250),
        # 'level_dim':              (155,250),
        'name':                   slide_name,
        'save_path':              h5_save_path}
    # attr = np.ones((100,2))
    attr_dict = { 'coords' : attr}
    
    asset_dict = {'coords' : coords}
    if not os.path.exists(h5_save_path):
        with h5py.File(h5_save_path,'a') as file:
            for key, val in asset_dict.items():
                data_shape = val.shape
                data_type = val.dtype 
                chunk_shape = (1, ) + data_shape[1:]
                max_shape = (None, ) + data_shape[1:]
                # print(key, data_shape, data_type)
                dset = file.create_dataset(key, shape = data_shape, dtype = data_type)
                dset[:] = val 
                for attr_key, attr_val in attr.items():
                    dset.attrs[attr_key] = attr_val

'''
    # print(coords)

    # print(slide_names)



# attr = {'patch_size' :            256, # To be considered...
#         'patch_level' :           0,
#         'downsample':             1,
#         'downsampled_level_dim' : (155,250),
#         'level_dim':              (155,250),
#         'name':                   "Harsh",
#         'save_path':              "./new_h5.h5"}
# # attr = np.ones((100,2))
# attr_dict = { 'coords' : attr}

# asset_dict = {'coords' : np.ones((100,2))}
# pathh = 'new_n.h5'
# with h5py.File(pathh,'a') as file:
#     for key, val in asset_dict.items():
#         data_shape = val.shape
#         data_type = val.dtype 
#         chunk_shape = (1, ) + data_shape[1:]
#         max_shape = (None, ) + data_shape[1:]
#         print(key, data_shape, data_type)
#         dset = file.create_dataset(key, shape = data_shape, dtype = data_type)
#         dset[:] = val 
#         for attr_key, attr_val in attr.items():
#             dset.attrs[attr_key] = attr_val

# with h5py.File(pathh,"r") as f:
#     # with h5py.File(new_h5, 'w') as w:
#     dset = f['coords']
#     print(dset,dset.attrs.keys())
#     # for att in dset.attrs.keys():
#     #     for key, val in att.attrs.items():
#     #         print(key, val)
'''