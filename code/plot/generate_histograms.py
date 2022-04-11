from argparse import ArgumentParser
import os, glob
from re import split
import matplotlib.pyplot as plt
import numpy as np
import PIL
import torchvision 
import torch
from tqdm import tqdm
from scipy import ndimage
from skimage import morphology
from skimage.filters import threshold_otsu
import cv2
from matplotlib.colors import hsv_to_rgb
import datetime

def parser():
    ap = ArgumentParser(add_help=False)
    ap.add_argument('path', type=str, help='Path to all model directories (must be called "fake_xy")')
    return ap

hparams = parser().parse_args()

class CustomDataSet(torch.utils.data.Dataset):
    def __init__(self, main_dir, transform):
        self.main_dir = main_dir
        self.transform = transform
        all_imgs = os.listdir(main_dir)
        self.total_imgs = all_imgs

    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, idx):
        img_loc = os.path.join(self.main_dir, self.total_imgs[idx])
        image = PIL.Image.open(img_loc).convert('HSV')
        tensor_image = self.transform(image)
        return tensor_image


if 'breakhis'.lower() in split('/', hparams.path)[-1].lower():
    start_ds = 'IDC'
    target_ds = 'BreaKHis'
    title = r"$ G_{IDC \rightarrow BreaKHis} $"
else:
    start_ds = 'BreaKHis'
    target_ds = 'IDC'
    title = r"$ G_{BreaKHis \rightarrow IDC} $"


# list directories for original and modified (per 10 epochs)
all_folders = [x[0] for x in os.walk(hparams.path)]
all_fake = [folder for folder in all_folders if 'modified' in folder]
all_orig = [folder for folder in all_folders if 'original' in folder]

# replace if interested in other epoch(s)
fake_models = [folder for folder in all_fake if 'l_150' in folder]

# dataloader transformations for dataset
img_sz = 96
test_transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize([img_sz, img_sz]),
    torchvision.transforms.ToTensor()
])

# create histograms from data path
bin_edges = np.arange(102)
def get_histogram(path):
    # extract data for original images
    dataset = CustomDataSet(path, transform=test_transforms)
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False, 
                                num_workers=4, drop_last=False)
    data = []
    for image_data in loader:
        data.append(image_data)
    data = torch.cat(data)

    # calculate mean value for original images
    data = data.detach().cpu().numpy()
    mean_data = np.mean(data, axis=0)
    mean_data = (100*mean_data).astype(int)

    # create histograms
    histos = []
    for channel_id in (0, 1, 2):
        histogram, bin_edges = np.histogram(
            mean_data[:, :, channel_id], bins=101, range=(0, 101)
        )
        histos.append(histogram)
    histos = np.asarray(histos)

    return histos

orig_histogram = get_histogram(all_orig[0])
orig2_histogram = get_histogram('Breakhis_IDC_100_2/histo_comparison/fake_{}/Model_150/original'.format(start_ds.lower()))

# extract histogram data for fake images
all_fake_histograms = []
all_model_idx = []
for model in tqdm(fake_models):
    dir_list = split('/', model)
    model_idx = int(''.join(filter(str.isdigit, str(dir_list[3]))))
    all_model_idx.append(str(model_idx))

    all_fake_histograms.append(get_histogram(model))

all_fake_histograms = np.asarray(all_fake_histograms)


plt.rcParams["font.family"] = "Helvetica"
fig, ax = plt.subplots(1,1, figsize = (12, 5))
fig.suptitle(title, fontsize=24)
ax.plot(bin_edges[0:-1], orig_histogram[0], 'k', label='target ({})'.format(target_ds), linewidth=2)
ax.plot(bin_edges[0:-1], orig2_histogram[0], 'tab:blue', linestyle='dotted', label='start ({})'.format(start_ds), linewidth=2)

for i,idx in enumerate(all_model_idx):
    if int(idx) == 150:
        ax.plot(bin_edges[0:-1], all_fake_histograms[i,0,:], 'tab:orange', linestyle='--', label='end (Fake {})'.format(target_ds), linewidth=2)

ax.set_xlabel('Hue', fontsize=17)
ax.set_ylabel("Pixels", fontsize=17)
ax.legend(fontsize=19, loc='upper center')
ax.set_ylim([0,100])
ax.set_xlim([10,90])
plt.setp(ax.get_xticklabels(), fontsize=16)
plt.setp(ax.get_yticklabels(), fontsize=16)
plt.subplots_adjust(left=0.01, right=0.99)
plt.savefig('histograms_fake_{}_150.png'.format(target_ds.lower()),bbox_inches='tight')
