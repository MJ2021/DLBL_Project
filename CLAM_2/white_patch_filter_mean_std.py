import pandas as pd 
import openslide as op
import matplotlib.pyplot as plt
import numpy as np
import glob
import os

#csv_path='/home/Drivessd2tb/dlbl_data/Mohit/FEATURES_FINAL/csv_files_without_white_filter/022365CN__20231104_094753.csv'
annot_csv_files=glob.glob('/home/Drivessd2tb/Mohit_Combined/csv_files_with_white_filter_from_patches_combined/*', recursive = True)
#annot_csv_files=glob.glob('/workspace/clam1/CLAM/rem_csv/*', recursive=True)
#print(annot_csv_files)
print(len(annot_csv_files))
for i in range(len(annot_csv_files)):
    csv_path = annot_csv_files[i]
    # print("CSV PATH IS ", csv_path)

    slide = csv_path.split('.')[0]
    slide_name = slide.split('/')[5]

    tiff_file_path = '/home/Drivessd2tb/dlbl_combined/' + slide_name + '.tiff'
    # print("Path is ", tiff_file_path)

    # tiff_file_path='/home/Drivessd2tb/dlbl_data/dlbl_tiff/' + p
    # print("Tiff File Path is ", tiff_file_path)

    save_path = '/home/Drivessd2tb/Mohit_Combined/csv_files_with_white_filter_from_patches_combined_210/' + slide_name + '.csv'

    # save_path1= '/home/ravi/Mohit/CLAM/white_filtered_csv_files/'+save_path
    if os.path.exists(save_path):
        print('done')
    else:
        print(f'{tiff_file_path} is in process')
        #if svs_file_path !='/workspace/hpv_tmh/CAIB-T00001462OP01B01P0101HE.svs':

        wsi= op.OpenSlide(tiff_file_path)
        df = pd.read_csv(csv_path)
        data_record = {'x':[],'y':[]}
        for i in range(len(df)):
            x = df['x'][i]
            y = df['y'][i]
            #print(f'x is {x} and y is {y}')
            level_zero_img= wsi.read_region((x,y), 0, (256,256)) 
            #converting in rgb
            level_zero_img_rgb=level_zero_img.convert('RGB')
            level_zero_img_np = np.array(level_zero_img_rgb)
            # print(level_zero_img_np.mean())
            if level_zero_img_np.mean()<210 and level_zero_img_np.mean()> 10 and  level_zero_img_np.std()>10:
                data_record['x'].append(x)
                data_record['y'].append(y)
            # else:
            #     # print('coords are dropped')
        df_tiff = pd.DataFrame(data_record)
        print("Original Packets = ", len(df))
        print("Packets Dropped = ", len(df) - len(df_tiff))

        #print(save_path1)
        df_tiff.to_csv(save_path,index=False)

        




'''
svs_file_path ='/workspace/hpv_tmh/CAIB-T00001462OP01B01P0101HE.svs'
csv_path='/workspace/clam1/CLAM/csv_tmc_pma_annot/CAIB-T00001462OP01B01P0101HE.csv'            
wsi= op.OpenSlide(svs_file_path)
df=pd.read_csv(csv_path)
data_record = {'dim1':[],'dim2':[]}
for i in range(len(df)):
    x=df['dim1'][i]
    y=df['dim2'][i]
                    
    level_zero_img= wsi.read_region((x,y), 0, (256,256)) 
                    
    level_zero_img_rgb=level_zero_img.convert('RGB')
    level_zero_img_np = np.array(level_zero_img_rgb)
    print(level_zero_img_np.mean())
    if level_zero_img_np.mean()<220 and level_zero_img_np.mean()> 10 and level_zero_img_np.std()>10:
        data_record['dim1'].append(x)
        data_record['dim2'].append(y)
    else:
        print('coords are dropped')
df_svs = pd.DataFrame(data_record)
        
df_svs.to_csv('gjhggd.csv',index=False)
    
'''
