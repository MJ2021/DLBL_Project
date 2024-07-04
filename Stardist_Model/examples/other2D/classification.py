from __future__ import print_function, unicode_literals, absolute_import, division
import sys
import numpy as np
import matplotlib
matplotlib.rcParams["image.interpolation"] = "nearest"
import matplotlib.pyplot as plt
# %matplotlib inline
# %config InlineBackend.figure_format = 'retina'

from glob import glob
import json
from tqdm import tqdm
from tifffile import imread
from csbdeep.utils import Path, normalize

from stardist import fill_label_holes, random_label_cmap, calculate_extents, gputools_available
from stardist.matching import matching, matching_dataset
from stardist.models import Config2D, StarDist2D, StarDistData2D
from stardist.utils import mask_to_categorical
from stardist.plot import render_label


np.random.seed(0)
lbl_cmap = random_label_cmap()
lbl_cmap_classes = matplotlib.cm.tab20
def plot_img_label(img, lbl, cls_dict, n_classes=2, img_title="image", lbl_title="label", cls_title="classes", **kwargs):
    c = mask_to_categorical(lbl, n_classes=n_classes, classes=cls_dict)
    res = np.zeros(lbl.shape, np.uint16)
    for i in range(1,c.shape[-1]):
        m = c[...,i]>0
        res[m] = i
    class_img = lbl_cmap_classes(res)
    class_img[...,:3][res==0] = 0 
    class_img[...,-1][res==0] = 1
    
    fig, (ai,al,ac) = plt.subplots(1,3, figsize=(17,7), gridspec_kw=dict(width_ratios=(1.,1,1)))
    im = ai.imshow(img, cmap='gray')
    #fig.colorbar(im, ax = ai)
    ai.set_title(img_title)    
    al.imshow(render_label(lbl, .8*normalize(img, clip=True), normalize_img=False, alpha_boundary=.8,cmap=lbl_cmap))
    al.set_title(lbl_title)
    ac.imshow(class_img)
    ac.imshow(render_label(res, .8*normalize(img, clip=True), normalize_img=False, alpha_boundary=.8, cmap=lbl_cmap_classes))
    ac.set_title(cls_title)
    plt.tight_layout()    
    for a in ai,al,ac:
        a.axis("off")
    return ai,al,ac

# set the number of object classes
n_classes = 6

img_arr = np.load('/home/ravi/Mohit/stardist/archive/data/images.npy')
ann_arr = np.load('/home/ravi/Mohit/stardist/archive/data/labels.npy')

seg_arr = ann_arr[:,:,:,0]
class_arr = ann_arr[:,:,:,1]

print(img_arr.shape)
print(seg_arr.shape)
print(class_arr.shape)

plt.imshow(seg_arr[0], interpolation='nearest')
plt.show()
plt.imshow(class_arr[0], interpolation='nearest')
plt.show()
plt.imshow(img_arr[0], interpolation='nearest')
plt.show()


def get_sample_prob(cls_dict):

    neu_count = 0
    eos_count = 0

    for key in cls_dict:

        val = cls_dict[key]
        
        if val == 1:
            neu_count += 1

        if val == 5:
            eos_count += 1

    tot_minor = neu_count + eos_count

    return tot_minor

# Create class dictionary

tuple_list = []
prob_list = []
idx_list = []

for i in tqdm(range(len(img_arr))):

    cls_dict = {}
    cur_img = img_arr[i]
    cur_seg = seg_arr[i]
    cur_class = class_arr[i]

    comb_arr = cur_seg + 1000*cur_class
    unique_comb = np.unique(comb_arr)

    for val in unique_comb:
        
        rem = val%1000
        val = val - rem
        quo = int(val/1000)
        cls_dict[rem] = quo

    prob = get_sample_prob(cls_dict)

    if(prob >= 5):
        prob_list.append(prob)
        idx_list.append(i)

    tuple_list.append((cur_img, cur_seg, cls_dict))

tot = sum(prob_list)
norm = [float(i)/tot for i in prob_list]        

ex_img, ex_seg, ex_cls = tuple_list[10]

ax = plot_img_label(ex_img,ex_seg,ex_cls, n_classes=n_classes)
for a in ax: a.axis("off")

# Adding 1500 extra samples based on oversampling
import random

extra_idx = random.choices(population=idx_list, weights=norm, k=1500)

for idx in extra_idx:

    tup = tuple_list[idx]
    tuple_list.append(tup)

print(len(tuple_list))

X, Y, C = tuple(zip(*tuple(tuple_list)))

assert len(X) == len(Y) == len(C)

n_channel = 1 if X[0].ndim == 2 else X[0].shape[-1]

axis_norm = (0,1)   # normalize channels independently
# axis_norm = (0,1,2) # normalize channels jointly
if n_channel > 1:
    print("Normalizing image channels %s." % ('jointly' if axis_norm is None or 2 in axis_norm else 'independently'))
    sys.stdout.flush()

X = [normalize(x,1,99.8,axis=axis_norm) for x in tqdm(X)]
Y = [fill_label_holes(y) for y in tqdm(Y)]

assert len(X) > 1, "not enough training data"
rng = np.random.RandomState(42)
ind = rng.permutation(len(X))
n_val = max(1, int(round(0.15 * len(ind))))
ind_train, ind_val = ind[:-n_val], ind[-n_val:]
X_val, Y_val, C_val = [X[i] for i in ind_val]  , [Y[i] for i in ind_val] , [C[i] for i in ind_val]
X_trn, Y_trn, C_trn = [X[i] for i in ind_train], [Y[i] for i in ind_train],  [C[i] for i in ind_train] 
print('number of images: %3d' % len(X))
print('- training:       %3d' % len(X_trn))
print('- validation:     %3d' % len(X_val))

i = min(8, len(X)-1)
img, lbl, cls = X[i], Y[i], C[i]
assert img.ndim in (2,3)
img = img if (img.ndim==2 or img.shape[-1]==3) else img[...,0]
plot_img_label(img, lbl, cls, n_classes)

print(Config2D.__doc__)

# 32 is a good default choice
n_rays = 32

# Use OpenCL-based computations for data generator during training (requires 'gputools')
use_gpu = True and gputools_available()

# Predict on subsampled grid for increased efficiency and larger field of view
grid = (2,2)

conf = Config2D (
    n_rays       = n_rays,
    grid         = grid,
    use_gpu      = use_gpu,
    n_channel_in = n_channel,
    n_classes    = n_classes,   # set the number of object classes
)
print(conf)
vars(conf)

if use_gpu:
    from csbdeep.utils.tf import limit_gpu_memory
    limit_gpu_memory(None, allow_growth=True)
    # alternatively, adjust as necessary: limit GPU memory to be used by TensorFlow to leave some to OpenCL-based computations
    # limit_gpu_memory(0.8)

model = StarDist2D(conf, name='stardist_multiclass', basedir='models')
model = StarDist2D(None, name='stardist_multiclass', basedir='models')

median_size = calculate_extents(list(Y), np.median)
fov = np.array(model._axes_tile_overlap('YX'))
print(f"median object size:      {median_size}")
print(f"network field of view :  {fov}")
if any(median_size > fov):
    print("WARNING: median object size larger than field of view of the neural network.")

def random_fliprot(img, mask): 
    assert img.ndim >= mask.ndim
    axes = tuple(range(mask.ndim))
    perm = tuple(np.random.permutation(axes))
    img = img.transpose(perm + tuple(range(mask.ndim, img.ndim))) 
    mask = mask.transpose(perm) 
    for ax in axes: 
        if np.random.rand() > 0.5:
            img = np.flip(img, axis=ax)
            mask = np.flip(mask, axis=ax)
    return img, mask 

def augmenter(x, y):
    """Augmentation of a single input/label image pair.
    x is an input image
    y is the corresponding ground-truth label image
    """
    #random flops and rotations
    x, y = random_fliprot(x, y)
    # add some gaussian noise
    sig = 0.02*np.random.uniform(0,1)
    x = x + sig*np.random.normal(0,1,x.shape)
    return x, y

# plot some augmented examples
img, lbl, cls = X[0], Y[0], C[0]
plot_img_label(img,lbl,cls, n_classes=n_classes)
for _ in range(3):
    img_aug, lbl_aug = augmenter(img,lbl)
    plot_img_label(img_aug,lbl_aug,cls, img_title="image augmented", lbl_title="label augmented", n_classes=n_classes)

model.train(X_trn,Y_trn, classes=C_trn, validation_data=(X_val,Y_val,C_val), augmenter=augmenter,
            epochs=5)

model.optimize_thresholds(X_val, Y_val)

i = 500
label, res = model.predict_instances(X_val[i], n_tiles=model._guess_n_tiles(X_val[i]))

# the class object ids are stored in the 'results' dict and correspond to the label ids in increasing order 

def class_from_res(res):
    cls_dict = dict((i+1,c) for i,c in enumerate(res['class_id']))
    return cls_dict

print(class_from_res(res))

plot_img_label(X_val[i], Y_val[i], C_val[i], n_classes=n_classes, lbl_title="GT")
plot_img_label(X_val[i], label, class_from_res(res), n_classes=n_classes, lbl_title="Pred")

# Testing on custom
from imageio import imread

def abspath(path):
    import os
    base_path = os.path.abspath(os.path.dirname(__file__))
    return os.path.join(base_path, path)

img = imread("/home/Drivessd2tb/Mohit_Combined/high_attention_patches_from_heatmap/ABC/000104CZ__20240628_091716/0.png")
img = normalize(img)
# img = X_val[500]

label, res = model.predict_instances(img, n_tiles=model._guess_n_tiles(img))
plot_img_label(img, label, class_from_res(res), n_classes=n_classes, lbl_title="Pred")

