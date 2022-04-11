from argparse import ArgumentParser
import os, glob
from re import split
import matplotlib.pyplot as plt
import numpy as np
import PIL
import torchvision 
import torch
from tqdm import tqdm
import matplotlib.font_manager

def parser():
    ap = ArgumentParser()
    ap.add_argument('path', type=str, 
                    help = 'Path to directories "fake_idc" and "fake_breakhis')
    return ap

hparams = parser().parse_args()

# Replace Path to generated images if needed

idc_ids = ['5', '126', '222', '319']        # fake idc
bh_ids = ['6', '63', '98', '126']           # fake breakhis

# fake idc
all_fake_idc = os.listdir(hparams.path + '/fake_idc/Model_150/modified')
all_orig_bh = os.listdir(hparams.path + '/fake_idc/Model_150/original')
# fake breakhis
all_fake_bh = os.listdir(hparams.path + '/fake_breakhis/Model_150/modified')
all_orig_idc = os.listdir(hparams.path + '/fake_breakhis/Model_150/original')

# get image names
f_idc = []
for img in all_fake_idc:
    img_id = split('_', img)[0]
    for i in idc_ids:
        if i == img_id:
            f_idc.append(img)

f_bh = []
for img in all_fake_bh:
    img_id = split('_', img)[0]
    for i in bh_ids:
        if i == img_id:
            f_bh.append(img)

# get image paths
fake_idc = [os.path.join(hparams.path+ '/fake_idc/Model_150/modified', pic) for pic in f_idc]
orig_bh = [os.path.join(hparams.path+ '/fake_idc/Model_150/original', pic) for pic in f_idc]

fake_bh = [os.path.join(hparams.path+ '/fake_breakhis/Model_150/modified', pic) for pic in f_bh]
orig_idc = [os.path.join(hparams.path+ '/fake_breakhis/Model_150/original', pic) for pic in f_bh]

print(os.path.join(hparams.path + '/fake_idc/Model_150/modified', f_idc[0]))


# plot images
plt.rcParams["font.family"] = "Helvetica"
fig,ax = plt.subplots(4,4, figsize = (7, 7))

for idx in range(len(fake_idc)):
    
    f_i = PIL.Image.open(fake_idc[idx]).convert("RGB")
    o_b = PIL.Image.open(orig_bh[idx]).convert("RGB")
    f_b = PIL.Image.open(fake_bh[idx]).convert("RGB")
    o_i = PIL.Image.open(orig_idc[idx]).convert("RGB")

    ax[idx,0].imshow(o_i)
    ax[idx,1].imshow(f_b)
    ax[idx,2].imshow(o_b)
    ax[idx,3].imshow(f_i)

ax[0,0].set_title("Original IDC\n", size=16)
ax[0,1].set_title("Fake BreaKHis\n", size=16)
plt.text(x=0.283 , y=1.01 , s= r"$ G_{IDC \rightarrow BreaKHis} $", fontsize=14, transform=fig.transFigure)
ax[0,2].set_title("Original BreaKHis\n", size=16)
ax[0,3].set_title("Fake IDC\n", size=16)
plt.text(x=0.805 , y=1.01 , s= r"$ G_{BreaKHis \rightarrow IDC}$", fontsize=14, transform=fig.transFigure)


for a in ax:
    for a2 in a:
        a2.set(frame_on=False)
        
        
plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
plt.xticks(color='w')
plt.yticks(color='w')
plt.subplots_adjust(left=0.001, right=0.999, bottom=0.001, top=0.999)
plt.tight_layout()
plt.savefig('example_images.png', bbox_inches='tight')