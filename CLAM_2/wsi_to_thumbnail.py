import openslide as op
import glob
import numpy as np

file= '/home/Drivessd2tb/dlbl_data/dlbl_tiff/005353CN__20240214_125335.tiff'
save_dir = '/home/Drivessd2tb/Mohit2_new/Thumbnails_new/'
input_dir = glob.glob('/home/Drivessd2tb/dlbl_tiff_new/*', recursive = True)
print(len(input_dir))

for i in range(len(input_dir)):

    wsi= op.OpenSlide(input_dir[i])
    dim=wsi.dimensions
    lvl=wsi.level_dimensions
    print('dim=',dim)
    print('lvl=',lvl)

    dwn_sample=wsi.level_downsamples
    # print(dwn_sample)

    prop=wsi.properties
    # print(prop)

    bst_dwn=wsi.get_best_level_for_downsample
    # print('bst',bst_dwn)

    slide = input_dir[i].split('.')[0]
    slide_name = slide.split('/')[4]

    thum=wsi.get_thumbnail(size=(dim[0]/512,dim[1]/512))
    thum=thum.save(save_dir + slide_name + '.png')