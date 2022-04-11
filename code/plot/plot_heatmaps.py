'''
This script handles the visualization of heatmaps.
It requires a folder structure of:

data/
  |- model1/
      |- hmaps/
      |- masks/
      |- origs/
  |- model2/
      |- hmaps/
      |- masks/
      |- origs/
  |- ...
  |- attn_01/
      |- hmaps/
      |- masks/
      |- origs/
  |- attn_02/
      |- hmaps/
      |- masks/
      |- origs/
  |- ...
  |- attn_12/
      |- hmaps/
      |- masks/
      |- origs/


containing the heatmaps, nuclei masks and original images. 
See main_heatmaps.py for the generation of the data.
'''
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import colors
import cv2
from termcolor import *
from scipy.ndimage.morphology import binary_erosion, binary_closing, binary_opening
import sys


HATCH_NUCLEI = '.'
HATCH_BG = 'x'


class BackgroundSegmentation(torch.nn.Module):

    def __init__(self, threshold=205/255, use_morphology=True, colour_channel_idx=3):
        super().__init__()
        self.threshold = threshold
        self.use_morphology = use_morphology
        self.colour_channel_idx = colour_channel_idx

    def forward(self, x):
        # ndarray sanity
        if isinstance(x, torch.Tensor):
            x_arr = x.cpu().detach().numpy()
        else:
            x_arr = np.array(x, dtype=np.float32)
        # thresholding for white background-blobs
        gray_img = x_arr.mean(axis=self.colour_channel_idx)
        #print(gray_img.max(), self.threshold)
        binary_img = x_arr.mean(axis=self.colour_channel_idx) > self.threshold
        # noise reduction & tight borders via erosion and opening
        if self.use_morphology:
            for i in range(binary_img.shape[0]):
                binary_img[i] = binary_opening(binary_img[i], structure=np.ones([3,3]))
        # Tensor conversion
        binary_img = torch.from_numpy(binary_img.astype(np.float32))
        if isinstance(x, torch.Tensor):
            binary_img = binary_img.to(x.device)

        binary_img = torch.cat([(1-binary_img)[:,None,:,:], binary_img[:,None,:,:]], dim=1)
        return binary_img.detach().cpu().numpy().astype(np.uint8)[0,1]


class Data:

    def __init__(self, root):
        self.root = root
        self.mdls = os.listdir(self.root)
        self.fs = {}
        self.bg_seg = BackgroundSegmentation()
        for m in self.mdls:
            self.fs[m] = {}
            for category in ('hmaps', 'origs', 'masks'):
                try:
                    self.fs[m][category] = sorted(os.listdir(os.path.join(self.root, m, category)))
                except:
                    pass

    def path(self, mdl, i, category='hmaps'):
        return os.path.join(self.root, mdl, category, self.fs[mdl][category][i])

    def load(self, mdl, i, category='hmaps'):
        data = np.load(self.path(mdl, i, category))
        if category == 'masks':
            data = (data  > 0).astype(np.uint8)
            bg = cv2.resize(self.bg_seg(self.load(mdl, i, 'origs')[None, :]), data.shape).astype(np.uint8)
            data = (data * (1-bg)).astype(np.uint8)
            data = (data, bg, (np.zeros(data.shape) - data - bg) >= 0)
        return data

    def len_mdl(self, mdl, category='hmaps'):
        return len(self.fs[mdl][category])


def plot(data, mdl, idx, category='hmaps', cmap='bwr'):
    if category == 'hmaps':
        h = data.load(mdl, idx, category)
        divnorm=colors.TwoSlopeNorm(vmin=min(-1e-8, h.min()), vcenter=0., vmax=max(1e-8, h.max()))
        nuclei, bg, tissue = map(lambda arr: cv2.resize(arr.astype(np.uint8), h.shape), data.load(mdl, idx, category='masks'))
        plt.contour(nuclei, colors='k')
        plt.contour(bg, colors='k')
        cs = plt.contourf(nuclei, colors='k', levels=2, hatches=[None, None, HATCH_NUCLEI], alpha=0.)
        cs = plt.contourf(bg, colors='k', levels=2, hatches=[None, None, HATCH_BG], alpha=0.)
        if category == 'hmaps':
            plt.imshow(h, norm=divnorm, cmap=cmap)
        else:
            plt.imshow(h, cmap=cmap)
    elif category == 'origs':
        o = data.load(mdl, idx, category)
        plt.imshow(o)
        nuclei, bg, tissue = map(lambda arr: cv2.resize(arr.astype(np.uint8), o.shape[:-1]), data.load(mdl, idx, category='masks'))
        plt.contour(nuclei, colors='k')
        plt.contour(bg, colors='k')
        plt.contourf(nuclei, colors='k', levels=2, hatches=[None, None, HATCH_NUCLEI], alpha=0.)
        plt.contourf(bg, colors='k', levels=2, hatches=[None, None, HATCH_BG], alpha=0.)
    else:
        raise ValueError(f'unknown category {category}')
    plt.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99)


if __name__ == '__main__':
    plt.rcParams["font.family"] = "Helvetica"
    data = Data('data')
    for m in [e for e in data.mdls if e[0] != 'a']:
        print(colored(m, 'green'))
        for i in [int(e) for e in sys.argv[3:]]:
            plot(data, m, i, sys.argv[1], cmap=sys.argv[2])
            plt.xticks([])
            plt.yticks([])
            fig = plt.gcf()
            fig.set_size_inches(10, 10)
            plt.savefig(f'{m}_{i}.png')
            plt.pause(0.05)
            plt.clf()

