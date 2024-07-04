import openslide as op
import glob
import numpy as np
import pandas as pd
import os

# file = '/home/Drivessd2tb/dlbl_combined/000104CZ__20240628_091716.tiff'
input_dir = '/home/Drivessd2tb/dlbl_combined/'
save_dir_high = '/home/Drivessd2tb/Mohit_Combined/high_attention_patches_from_heatmap/'
save_dir_low ='/home/Drivessd2tb/Mohit_Combined/low_attention_patches_from_heatmap/'
csv_path = '/home/Drivessd2tb/Mohit_Combined/csv_files_from_heatmap_h5/GCB/011487CZ__20240628_091310.csv'

slide = csv_path.split('.')[0]
slide_name = slide.split('/')[6]
class_name = slide.split('/')[5]
file = input_dir + slide_name + '.tiff'

wsi= op.OpenSlide(file)
dim=wsi.dimensions
lvl=wsi.level_dimensions
print('dim=',dim)
print('lvl=',lvl)

save_dir_low += (class_name + '/')
save_dir_high += (class_name + '/')

directory_path_low = os.path.join(save_dir_low, slide_name)
directory_path_high = os.path.join(save_dir_high, slide_name)
if not os.path.exists(directory_path_low):
    os.makedirs(directory_path_low)

if not os.path.exists(directory_path_high):
    os.makedirs(directory_path_high)

df = pd.read_csv(csv_path)
sorted_df = df.sort_values(by='attention', ascending=True)
sorted_df.reset_index(drop=True, inplace=True)

for i in range(10):
    x = sorted_df['x'][i]
    y = sorted_df['y'][i]
    print(sorted_df['attention'][i])
    level_zero_img= wsi.read_region((x,y), 0, (256,256)) 
    #converting in rgb
    level_zero_img_rgb=level_zero_img.convert('RGB')
    level_zero_img_rgb = level_zero_img_rgb.save(directory_path_low + '/' + str(i) + '.png')

sorted_df = df.sort_values(by='attention', ascending=False)
sorted_df.reset_index(drop=True, inplace=True)

for i in range(10):
    x = sorted_df['x'][i]
    y = sorted_df['y'][i]
    print(sorted_df['attention'][i])
    level_zero_img= wsi.read_region((x,y), 0, (256,256)) 
    #converting in rgb
    level_zero_img_rgb=level_zero_img.convert('RGB')
    level_zero_img_rgb = level_zero_img_rgb.save(directory_path_high + '/' + str(i)  + '.png')
