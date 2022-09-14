from typing import Callable
import torch
import pytorch_lightning as pl
from matplotlib import colors
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib import colors
import seaborn as sns
import torch.nn.functional as F
from torchvision import transforms
import os
import json
import cv2
import imantics
from torch.utils.data import DataLoader, Dataset
from captum.attr import *
from captum.attr import visualization as viz
from captum.attr._utils.attribution import GradientAttribution
from argparse import ArgumentParser
import sys
import numpy as np
from sklearn.metrics import roc_auc_score
import torchmetrics.functional as F_metrics
from tqdm import tqdm
# script can be called from any directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import main_patho_lightning
import main_segmentation
import utils
from pytorch_LRP import resnet
from pytorch_LRP import layers as layers_LRP
from pytorch_LRP.transformer.vit import *
from copy import deepcopy
import gashis
from scipy.ndimage import gaussian_filter
from scipy.stats import pointbiserialr, pearsonr
from scipy.ndimage.morphology import binary_opening, binary_erosion

# needed during itereation of dataloader
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


def gen_visualizations(model, resize=None, max_samples=None):
    to_class = lambda t: ((t[:,0,:,:] - t[:,1,:,:]) < 0).type(torch.float32)
    Y_pred = []
    X = []
    with torch.no_grad():
        # setup evaluation settings
        model = model.eval()
        dloader = model.test_dataloader()
        if max_samples is None:
            max_samples = len(dloader)
        color = None
        for b_idx, batch in enumerate(tqdm(dloader, total=max_samples)):
            if max_samples is not None and b_idx >= max_samples:
                break
            # predict segmentation
            imgs, segs = map(lambda t: t.to(model.device), batch)
            if resize is not None:
                imgs, segs = map(lambda t: transforms.functional.resize(t, resize), [imgs, segs])
            preds = to_class(model(imgs))
            Y_pred.append(preds.cpu().numpy())
            X.append(imgs.cpu().numpy())
    return np.concatenate(Y_pred), np.concatenate(X)


def get_seg_masks(model_nuclei=None, model_bg=None, model_tissue=None, max_samples=None, resize=None, keep_origs=False):
    masks = []
    if keep_origs:
        origs = []
    for model in (model_nuclei, model_bg, model_tissue):
        if model is not None:
            model.eval()
            model = model.cuda()
            m, o = gen_visualizations(model, resize=resize, max_samples=max_samples)
            masks.append(m)
            if keep_origs:
                origs.append(o)
            model.cpu()
        else:
            masks.append(None)
            if keep_origs:
                origs.append(None)
    result = (masks, origs) if keep_origs else masks
    return result


def integrated_top_K(R, F, step_size=0.05):
    R_sorted = np.argsort(-R.reshape(np.prod(R.shape))).reshape(R.shape)
    K = F.sum()
    step = K * step_size
    steps = np.arange(step, K+1, step=step)
    curve = np.zeros(int(1/step_size))
    for i, k in enumerate(steps):
        itc = ((R_sorted <= k) * F).sum()
        curve[i] = itc / k
    return curve, K / np.prod(F.shape)


def mass_accuracy(R, F):
    return (R * F).sum() / R.sum()


def pool_avg(R):
    if isinstance(R, np.ndarray):
        R = torch.from_numpy(R)
    return torch.mean(R, dim=0)


def pool_avg_abs(R):
    return torch.abs(pool_avg(R))


def pool_avg_max(R):
    return pool_thresh(pool_avg(R), 0)


def pool_thresh(R, thresh=0):
    if isinstance(R, np.ndarray):
        R = torch.from_numpy(R)
    return torch.maximum(torch.ones(R.shape) * thresh, R)


def pool_max_norm(R):
    if isinstance(R, np.ndarray):
        R = torch.from_numpy(R)
    return torch.max(torch.abs(R), dim=0)[0]


def pool_pos_L2_norm_sq(R):
    if isinstance(R, np.ndarray):
        R = torch.from_numpy(R)
    return pool_L2_norm_sq(pool_thresh(R, 0))


def pool_L2_norm_sq(R):
    if isinstance(R, np.ndarray):
        R = torch.from_numpy(R)
    return torch.sum(R**2, dim=0)


POOL_FUNCTIOS = {
    'pool_avg': pool_avg,
    'pool_avg_abs': pool_avg_abs,
    'pool_avg_max': pool_avg_max,
    'pool_max_norm': pool_max_norm,
    'pool_pos_L2_norm_sq': pool_pos_L2_norm_sq,
    'pool_L2_norm_sq': pool_L2_norm_sq
}


def pearson_biserial_r(Y_hat, Y):
    assert all([e in [0, 1] for e in set(Y.flatten().tolist())]), "Y aught to be array-like booleans"
    if not isinstance(Y_hat, torch.Tensor):
        Y_hat = torch.tensor(Y_hat)
    if not isinstance(Y, torch.Tensor):
        Y = torch.tensor(Y)
    Y_hat, Y = map(lambda t: t.type(torch.float32), [Y_hat, Y])
    return pearson_r(torch.flatten(Y_hat, start_dim=1), torch.flatten(Y, start_dim=1))


def pearson_r(Y_hat, Y):
    r = torch.empty(len(Y_hat), dtype=float)
    for i, (y_hat, y) in enumerate(zip(Y_hat, Y)):
        r[i] = F_metrics.pearson_corrcoef(y_hat, y)
    return r


def rank_accuracy(R, seg_mask, K=None):
    if K is None:
        K = seg_mask.sum()
    R_K = (np.argsort(-R.reshape(np.prod(R.shape))) <= K).reshape(R.shape)
    return (R_K * seg_mask).sum() / min(K, seg_mask.sum())


def rank_accuracy_bbox(hmap, bbox=((32,32), (64, 64)), K=32**2):
    seg_mask = torch.zeros(hmap.shape)
    tl, br = bbox
    seg_mask[tl[0]:br[0], tl[1]:br[1]] = 1
    return rank_accuracy(hmap, seg_mask, K=K)


def rank_accuracy_curve(Rs, seg_masks, K_min=0, K_max=None, n_steps=20, f_pool=lambda x:x):
    """
    Calculates the curve of rank accuracies in a specified interval.

    :param Rs: relevance maps
    :param seg_masks: segmentation masks
    :param K_min: lower bound on K
    :param K_max: upper bound on K
    :param n_steps: number of steps to take from K_min to K_max
    :f_pool: pooling function, default: identity
    :return: curve of rank accuracy scores as a tuple of K-values and accuracy scores
    """
    if K_max is None:
        K_max = np.product(Rs.shape)
    ks = np.arange(K_min, K_max-(K_max % n_steps), n_steps)
    if (K_max % n_steps) != 0:
        ks = np.concatenate([ks, [K_max]])
    rank_accs = np.zeros(len(ks))
    for i, K in enumerate(ks):
        rank_accs[i] = rank_accuracy(Rs, seg_masks, K, f_pool)
    return ks, rank_accs


# def integrated_top_K(Rs, seg_masks, K_min=0, K_max=None, n_steps=20, f_pool=lambda x:x):
#     """
#     the area under the top-K/ rank accuracy curve (discretized)
#     """
#     ks, rank_accs = rank_accuracy_curve(Rs, seg_masks, K_min, K_max, n_steps, f_pool)
#     return rank_accs.sum(axis=0) / len(ks)
# 
def plot_rank_accuracy_curve(Rs, seg_masks, K_min=0, K_max=None, n_steps=20, f_pool=lambda x:x, ax=None):
    """
    plots the rank accuracies in a specified interval, see :meth:`code.main_heatmaps.rank_accuracy_curve`

    :param ax: axis to plot to, if None a new axis and figure will be generated
    :params kwargs: args passed to :meth:`code.main_heatmaps.rank_accuracy_curve`
    """
    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=[2,2])
    steps, rank_accs = rank_accuracy_curve(Rs=Rs, seg_masks=seg_masks, K_min=K_min, K_max=K_max, n_steps=n_steps, f_pool=f_pool)
    ax.plot(steps, rank_accs, label=rf'Rank Accuracies for $k \in [{K_min}, {K_max}]$')
    plt.legend()
    sns.despine()

def split_segs(mask):
    """
    splits a segmentation mask into individual, disjoint masks

    :param mask: segmentation mask of shape (H, W) containing n disjoint areas
    :return: array of shape (n, H, W) containing segmentation mask for individual areas
    """
    polys = imantics.Mask(mask).polygons()
    new_masks = np.array([imantics.Polygons([p]).mask(*mask.shape).array for p in polys])
    return new_masks


def score_areas(mask, R):
    """
    splits a segmentation mask into individual, disjoint masks and assigns relevance scores to those areas.

    :param mask: segmentation mask of shape (H, W) with n disjoint areas
    :param R: relevance heatmap of shape (H, W)
    :return: weighted segemntation masks for individual areas, of shape (n, H, W)
    """
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().detach().numpy()
    if isinstance(R, torch.Tensor):
        R = R.cpu().detach().numpy()
    new_masks = split_segs(mask)
    R_total = R.sum()
    R_vals = np.array([(m*R).sum() / R_total for m in new_masks])
    return new_masks * R_vals[:,None, None], R_total


def plot_scored_areas(scores, original_img=None, fname_out=None):
    """
    generates a plot for the scored areas

    :param score: weighted segmentation maps for n disjoint areas, shape of (n, H, W)
    :param original_image: the input image, shape of (H, W, 3)
    :param fname_out: path to the output file
    """
    # display the original image in background
    if not original_img is None:
        plt.imshow(original_img)
    # display individual areas
    vmin_global, vmax_global = scores.min(), scores.max()
    divnorm=colors.TwoSlopeNorm(vmin=min(-1e-8, vmin_global), vcenter=0., vmax=max(1e-8, vmax_global))
    for m in scores:
        f_alpha = 3/7 if original_img is not None else 1.0
        plt.imshow(m, cmap='RdYlGn', norm=divnorm, alpha=f_alpha*(m>0))
    # save polot if a filename was specified
    if fname_out is not None:
        if not os.path.isdir(os.path.dirname(fname_out)):
            os.makedirs(os.path.dirname(fname_out))
        plt.savefig(fname_out)
        # closing figure and axis after the plot has been saved
        plt.cla()
        plt.clf()


def write_score_plots(masks, Rs, fnames, origs, verbose=False):
    """
    This method calculates the area specific relevance scores and writes the resulting plots.

    :param mask: segmentation masks, shape: (N, H, W)
    :param Rs: Relevance-scores, shape: (N, H, W)
    :param fnames: output file names for plots, shape: (N,)
    :param origs: input images for the plots background, shape: (N, H, W, 3)
    :param verbose: if this flag is set, the method will output a progress bar and some info while running
    """
    if verbose:
        print('Generating aerial score plots')
        iterator = tqdm(zip(masks, Rs, fnames, origs), total=min(len(masks), len(Rs), len(fnames), len(origs)))
    else:
        iterator = zip(masks, Rs, fnames, origs)
    for m, R, f, x_in in iterator:
        new_masks, R_total = score_areas(m, R)
        plot_scored_areas(new_masks, original_img=x_in, fname_out=f)


class BackgroundSegmentation(torch.nn.Module):

    def __init__(self, threshold=205/255, use_morphology=True, colour_channel_idx=1):
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
        binary_img = 1.0 * (x_arr.mean(axis=self.colour_channel_idx) > self.threshold)
        # noise reduction & tight borders via erosion and opening
        if self.use_morphology:
            for i in range(binary_img.shape[0]):
                binary_img[i] = binary_opening(binary_img[i], structure=np.ones([3, 3]))
        # Tensor conversion
        binary_img = torch.from_numpy(binary_img.astype(np.float32))
        if isinstance(x, torch.Tensor):
            binary_img = binary_img.to(x.device)
        binary_img = torch.cat([(1-binary_img)[:,None,:,:], binary_img[:,None,:,:]], dim=1)
        return binary_img


class TissueSegmentation(torch.nn.Module):

    def __init__(self, bg_seg, cell_seg):
        super().__init__()
        self.bg_seg = bg_seg
        self.cell_seg = cell_seg

    def forward(self, x):
        _1 = torch.ones(1, *x.shape[-2:], dtype=torch.float32, device=x.device)
        _0 = torch.zeros(1, *x.shape[-2:], dtype=torch.float32, device=x.device)
        # background extraction
        bg = self.bg_seg(x)
        bg = ((bg[:,0,:,:] - bg[:,1,:,:]) < 0) * _1
        # cell extraction
        cells = self.cell_seg(x)
        cells = ((cells[:,0,:,:] - cells[:,1,:,:]) < 0) * _1
        # tissue is implicitly detected via abscence of cells and background
        tissue = torch.maximum((_1 - cells - bg), _0)
        return torch.cat([(1-tissue)[:,None,:,:], tissue[:,None,:,:]], dim=1)


class DiskMemoryList:
    """
    Storing all ~32K PCam for models with large Relevance-Maps isn't tractable w.r.t. memory consumption.
    Hence storing and loading elements from disc is the way to go.

    .. warning:
        While this data-object does not consume much RAM memory, each append and __getitem__ call induces I/O operations.
    """

    def __init__(self, folder, iterable=None, dtype=np.ndarray, writer=np.save, loader=np.load, suffix='.npy'):
        self.__folder = folder
        self.__file_names = []
        self.__N = 0
        self.__dtype = dtype
        self.__writer = writer
        self.__loader = loader
        self.__suffix = suffix
        if not os.path.isdir(folder):
            os.makedirs(folder)
        if not len(os.listdir(folder)) == 0:
            self.__N = len(os.listdir(folder))
            for i in range(len(self)):
                f_name = self.__file_name(i)
                assert os.path.isfile(f_name), f"non empty folder at {folder}, but {f_name} does not exist."
                self.__file_names.append(f_name)
        if iterable is not None:
            for element in iterable:
                self.append(element)

    def append(self, element):
        assert isinstance(element, self.__dtype)
        file_name = self.__file_name(self.__N)
        self.__writer(file_name, element)
        self.__file_names.append(file_name)
        if len(str(self.__N-1)) - len(str(self.__N)) < 0:
            for i in range(len(self)):
                self.__update_name(i)
        self.__N += 1

    def delete_all(self):
        for i in range(len(self)):
            f_name = self.__file_name(i)
            os.remove(f_name)
        self.__N = 0
        self.__file_names = []

    def __iadd__(self, e):
        """This method allows for the += operator to append multiple elements at once"""
        try:
            for element in e:
                self.append(element)
        except:
            self.append(e)
        return self

    def __len__(self):
        return self.__N

    def __getitem__(self, i):
        f_name = self.__file_name(i)
        element = self.__loader(f_name)
        return element

    def __update_name(self, i):
        f_name = self.__file_names[i]
        new_f_name = self.__file_name(i)
        os.rename(f_name, new_f_name)
        self.__file_names[i] = new_f_name

    def __file_name(self, i):
        leading_zeros = ( len(str(len(self))) - len(str(i)) ) * '0'
        return os.path.join(self.__folder, f'{leading_zeros}{i}{self.__suffix}')

    def __iter__(self):
        self.__iteration_idx = 0
        return self

    def __next__(self):
        if self.__iteration_idx >= len(self):
            self.__iteration_idx = 0
            raise StopIteration()
        element = self[self.__iteration_idx]
        self.__iteration_idx += 1
        return element


class HeatMapGenerator:
    """This class generates the heatmaps, obtained from specified XAI methods."""

    def __init__(self, model, data, f_xai, dir_out, device=torch.device('cuda'), batch_size=32, baselines=None, debug=False,
                 ig_steps=50, nt_samples=5, pool=pool_avg_max, out_shape=[96,96], trafos=torch.nn.Identity(), 
                 filter_fn=lambda x: True):
        """
        :param model: The Model to evaluate
        :param data: The test dataset split
        :param f_xai: The XAI method, responsible to generate heatmaps
        :param device: The GPU/ CPU device to use
        :param batch_size: The size of the mini batch (NOTE: carefull, IntGrad may need a lot of memory)
        """
        self.device = device
        self.model = model
        self.data = data
        self.f_xai = f_xai
        self.batch_size = batch_size
        self.baselines = baselines
        self.debug = debug
        self.ig_steps = ig_steps
        self.nt_samples = nt_samples
        self.pool = pool
        self.out_shape = out_shape
        self.__dir_out = dir_out
        self.trafos = trafos
        self.filter_fn = filter_fn

    def all_heatmaps(self, y_class=1, max_samples=None, target=None):
        """
        :param f_mask: a functional/ callable object that generates segmentation masks for a mini batch
        :param y_class: only samples of this class annotation will be considered
        :param target: the class to evaluate for
        :param pool_mass: the pooling method usde for mass accuracy
        :param pool_rank: the pooling method used for rank accuracy
        :return: mass accuracy, rank accuracy, heat maps, original images, scores, annotated class labels
        """
        d_loader = DataLoader(self.data, num_workers=4, batch_size=self.batch_size, drop_last=False)
        #mass_accs, rank_accs, h_maps, origs, y_preds, ys = [], [], [], [], [], []
        h_maps, origs, y_preds, ys = [], [], [], []
        p_bar = tqdm(total=len(d_loader))
        self.model.eval()
        self.model.to(self.device)
        # N = len(self.data)
        # C, H, W = self.data[0][0].shape
        h_maps = DiskMemoryList(folder=os.path.join(self.__dir_out, 'h_maps'), dtype=np.ndarray, writer=np.save, 
                                loader=np.load, suffix='.npy') #np.zeros((N, C, H, W), dtype=np.float32)
        h_maps.delete_all()
        origs = DiskMemoryList(folder=os.path.join(self.__dir_out, 'origs'), dtype=np.ndarray, writer=np.save, 
                               loader=np.load, suffix='.npy') #np.zeros((N, C, H, W), dtype=np.float32)
        origs.delete_all()
        ys = []
        y_preds = []
        for i, (x, y) in enumerate(d_loader):
            # no exceeding the max amount of samples specified
            if max_samples is not None and len(origs) >= max_samples:
                break
            # only samples with segmentation masks are relevant, hence filtering for the class if specified
            if y_class is not None:
                if (y == y_class).sum() == 0:
                    # updating the progress bar
                    p_bar.update(1)
                    continue
                x, y = x[y == y_class], y[y == y_class]
            # filtering to criterion
            fltr_mask = [self.filter_fn(x_i) for x_i in x]
            if sum(fltr_mask) == 0:
                p_bar.update(1)
                print([torch.sum(filter_bg(x[IDX][None,:])[:,1]) for IDX in range(len(x))])
                print(torch.sum(filter_bg(x)[:,1]))
                continue
            x, y = x[fltr_mask], y[fltr_mask]
            # moving input to GPU/ models device
            x = x.to(self.device)
            # calculating probabilities/ scores
            y_pred = F.softmax(self.model(self.trafos(x)), dim=1)
            # heatmap generation
            lbl = target if target is not None else torch.argmax(y_pred, dim=1)
            h_map = self.explain(x, lbl)
            # apply pooling
            h_map = [self.pool(h.transpose(2,0,1)).detach().cpu().numpy() for h in h_map]
            # storing the unpooled heatmaps
            h_maps += h_map
            # updating the progress bar
            p_bar.update(1)
            # storing original images for visualization
            #x = transforms.functional.resize(x, self.out_shape)
            to_np = lambda t: t.detach().cpu().numpy()
            x, y, y_pred = map(to_np, [x, y, y_pred])
            ys.append(y)
            y_preds.append(y_pred)
            origs += [x[i].transpose(1, 2, 0) for i in range(len(x))]
        p_bar.close()
        # gather results, casting to numpy arrays
        #ys = np.concatenate(ys)
        #y_preds = np.vstack(y_preds)
        return h_maps, origs, y_preds, ys

    def explain(self, x_in, target):
        """
        :param x_in: The mini batch of input images
        :param target: The class(es) to calculate the heatmap for
        :return: A list of heatmaps w.r.t. the XAI method definedn in self.f_xai
        """
        x_in = self.trafos(x_in)
        self.model.zero_grad()
        # gradient based methods
        if isinstance(self.f_xai, GradientAttribution) or isinstance(self.f_xai, NoiseTunnel):
            # gather baselines
            if self.baselines is None:
                baselines = torch.zeros(x_in.shape).to(self.device)
            else:
                baselines = torch.ones(x_in.shape).to(self.device) * torch.from_numpy(self.baselines).to(self.device)
            # gather attributes
            if isinstance(self.f_xai, IntegratedGradients):
                attributes = self.f_xai.attribute(x_in, target=target, internal_batch_size=self.batch_size,
                                                  baselines=baselines, n_steps=self.ig_steps)
            elif isinstance(self.f_xai, NoiseTunnel):
                attributes = self.f_xai.attribute(x_in, target=target, internal_batch_size=self.nt_samples*self.batch_size,
                                                  baselines=baselines, nt_samples=self.nt_samples, n_steps=self.ig_steps)
            else:
                attributes = self.f_xai.attribute(x_in, target=target, baselines=baselines)
        # LRP/ RAP / Attention methods 
        elif isinstance(self.f_xai, RAP_LRP) or isinstance(self.f_xai, AttentionMap):
            attributes = self.f_xai(x_in, target)
        else:
            raise NotImplementedError(f'Method {type(self.f_xai)} not supported.')
        if not isinstance(attributes, np.ndarray):
            attributes = attributes.detach().cpu()
            attributes = attributes.numpy()
        # reshaping the heatmaps to (H, W, C) for display
        h_maps = [attributes[i].transpose(1, 2, 0) for i in range(len(attributes))]
        return h_maps


class SampleDset(Dataset):
    """Wrapper of a list/ array of images for a pytorch Dataset"""

    def __init__(self, samples, trafo=lambda x: x):
        self.samples = samples
        self.trafo = trafo

    def __getitem__(self, idx):
        return self.trafo(self.samples[idx]), np.zeros(self.samples[idx].shape)

    def __len__(self):
        return len(self.samples)


class RAP_LRP:

    def __init__(self, model, device=torch.device('cpu'), n_classes=2, method='RAP'):
        self.device = device
        self.model = model
        self.n_classes = n_classes
        self.method = method

    def __call__(self, x_in, y_class=None):
        # output without softmax
        self.model.eval()
        logit = self.model(x_in)
        self.model.zero_grad()
        # not asking for a specific class => use models predicted class
        if y_class is None:
            pred = torch.argmax(logit.data, dim=1)
        # asking for specfic classes
        else:
            # asking for a sequence of classes, e.g. GT-annotations
            if isinstance(y_class, np.ndarray):
                y_class = torch.from_numpy(y_class)
            if isinstance(y_class, torch.Tensor):
                if len(y_class.shape) == 1:
                    if y_class.shape[0] == x_in.shape[0]:
                        y_class = y_class[:, None]
                    else:
                        raise ValueError('Either submit a class for each input or one class for all.')
                elif y_class.shape != (len(x_in), 1):
                        raise ValueError('Either submit a class for each input or one class for all.')
            # asking for a single class applying to all input
            else:
                y_class = torch.ones((len(x_in), 1)) * y_class
            pred = y_class
        pred = pred.cpu()
        # generating one hot encodings
        T = pred == torch.arange(self.n_classes).type(torch.FloatTensor)
        T = torch.autograd.Variable(T).to(self.device)
        if isinstance(self.model, VisionTransformerConvStem) or isinstance(self.model, VisionTransformer):
            torch.sum(logit * T).backward()
            #print('register backward')
        # RAP 
        if self.method == 'RAP':
            scores = self.model.RAP_relprop(R=T)
        # LRP
        elif self.method == 'LRP':
            scores = self.model.relprop(R=T, alpha=2/3)
        else:
            raise NotImplementedError(f'unknown method {self.method}')
        # scores = rap.sum(dim=1, keepdim=True)
        scores = scores.detach().cpu().numpy()
        # return results
        return scores


class AttentionMap:
    DEFAULT_APPLIES = {
        'vit': lambda m, x, P, S_in, interpolation: m.get_attn_maps(x, P=P, S_in=S_in, interpolation=interpolation),
    }

    def __init__(self, model, f_model_apply='vit', P=16, S_in=224, interpolation='bicubic', n_head=-1):
        self.model = model.model
        self.type_model = f_model_apply
        print(f_model_apply)
        if not f_model_apply in self.DEFAULT_APPLIES.keys():
            assert isinstance(f_model_apply, Callable), 'submit a function or arch of the ViT'
            self.f_model_apply = f_model_apply
        else:
            self.f_model_apply = self.DEFAULT_APPLIES[f_model_apply]
        self.P = P
        self.S_in = S_in
        self.interpolation = interpolation
        self.n_head = n_head

    def __call__(self, x_in, y_class=None):
        self.model.eval()
        # attention maps are not reliant on gradients or backprop-hooks => speeding up the inference with no_grad environment
        with torch.no_grad():
            amap = self.f_model_apply(self.model, x_in, self.P, self.S_in, self.interpolation)
        hm = amap[:, self.n_head, :, :][:,None,:,:]
        # cpu/ numpy array output for memory efficiency
        hm = hm.detach().cpu().numpy()
        return hm


def save_plots(h_maps, origs, hparams, seg_masks=None, dir_suffix='', scores=None, gt=None):
    # sanity
    base_dir = os.path.join(hparams.output_path, dir_suffix)
    hm_dir = os.path.join(base_dir, 'plots')
    if not os.path.isdir(base_dir):
        os.makedirs(base_dir)
    if not os.path.isdir(hm_dir):
        os.makedirs(hm_dir)
    # writing heatmaps
    print('writing heatmap images')
    label_str = ['benign', 'cancer']
    for i, (hm, original_image) in tqdm(enumerate(zip(h_maps, origs)), total=len(h_maps)):
        # if there arte segmentation maps 4 plots will be shown
        if seg_masks is not None:
            fig, ((ax_orig, ax_seg),(ax_hmap, ax_itsct)) = plt.subplots(2, 2, figsize=[4, 4])
        # otherwise there will only be a plot of the original image and corresponding relevance or attention map
        else:
            fig, (ax_orig, ax_hmap) = plt.subplots(2, 1, figsize=[2, 4])
        # annotate prediction if possible
        title = ''
        if scores is not None:
            pred_cls = np.argmax(scores[i])
            title += r'$y_{pred} = %d' % (pred_cls,) + ' '
        if gt is not None:
            gt_cls = gt[i]
            title += r'$y = %d' % (gt_cls,)
        fig.suptitle(title)
        # displaying the original image and relevence or attention map
        hm_norm = colors.CenteredNorm()
        ax_orig.imshow(original_image)
        ax_orig.set_title('original')
        ax_hmap.imshow(hm, cmap='RdYlGn', norm=hm_norm)
        ax_hmap.set_title('heat map')
        # displaying the segmented area, aswell as the intersection of relevence/ attention and segementation
        if seg_masks is not None:
            seg_mask = seg_masks[i].squeeze()
            ax_seg.imshow(original_image)
            ax_seg.imshow(seg_mask, cmap='Greens', alpha=0.3*seg_mask)
            ax_seg.set_title('segmentation')
            ax_itsct.imshow(cv2.resize(original_image, hm.shape[:2][::-1]))
            seg_mask = cv2.resize(seg_mask, hm.shape[:2][::-1])
            ax_itsct.imshow(seg_mask*hm, cmap='Greens', alpha=0.3*seg_mask)
            ax_itsct.set_title('intersection')
        plt.tight_layout()
        plt.savefig(os.path.join(hm_dir, f'{i}.png'), dpi=80)
        plt.close(fig)
    # plotting the rank accuracies
    plot_rank_accuracy_curve(Rs=h_maps, seg_masks=np.array([cv2.resize(m, h_maps[i].shape[:2][::-1]) for i, m in enumerate(seg_masks)]))
    plt.savefig(os.path.join(base_dir, 'rank_accuracies.png'), dpi=80)


def get_f_xai(hparams, model):
    if hparams.xai_method == 'IntGrad':
        f_xai = IntegratedGradients(model)
    elif hparams.xai_method == 'IntGradNoiseTunnel':
        f_xai = NoiseTunnel(IntegratedGradients(model))
    elif hparams.xai_method == 'RAP':
        #assert hparams.arch in [f'resnet{n}' for n in [18, 34, 50, 101, 152]], "RAP was only implemented for the resnet arches."
        f_xai = RAP_LRP(model.model, device=torch.device('cuda'), method='RAP')
    elif hparams.xai_method == 'LRP':
        #assert hparams.arch in [f'resnet{n}' for n in [18, 34, 50, 101, 152]], "LRP was only implemented for the resnet arches."
        f_xai = RAP_LRP(model.model, device=torch.device('cuda'), method='LRP')
    elif hparams.xai_method == 'DeepLift':
        f_xai = DeepLift(model)
    elif hparams.xai_method == 'GradShap':
        f_xai = GradientShap(model)
    elif hparams.xai_method == 'Occlusion':
        f_xai = Occlusion(model)
    elif hparams.xai_method == 'Attention':
        f_xai = AttentionMap(model, f_model_apply='vit', n_head=hparams.n_head)
    else:
        raise NotImplementedError(f'Unsupported method {hparams.xai_method}, plese choose from (IntGrad, IntGradNoiseTunnel)')
    return f_xai


def get_data(hparams, test_transforms=transforms.Compose([transforms.ToTensor(), transforms.Resize([96, 96])])):
    # load the dataset and baseline references for gradient based XAI methods
    baselines = None
    if hparams.dataset == 'PCAM':
        file_dir = os.path.dirname(os.path.abspath(__file__))
        if os.path.isdir(os.path.join(file_dir, 'mean_imgs')):
            if os.path.isfile(os.path.join(file_dir, 'mean_imgs', 'mean_pcam.npy')): 
                baselines = np.load(os.path.join(file_dir, 'mean_imgs', 'mean_pcam.npy'))
                baselines = baselines.resize([baselines.shape[0], *hparams.img_shape])
                print(f'using {os.path.join(file_dir, "mean_imgs", "mean_pcam.npy")} as baselines if IntGrad was selected.')
    return baselines


def get_seg_models(hparams, data):
    pths = hparams.model_nuclei, hparams.model_tubule, hparams.model_epithelium
    arches = hparams.arch_nuclei, hparams.arch_tubule, hparams.arch_epithelium
    args = deepcopy(hparams)
    models = []
    for p, arch in zip(pths, arches):
        if p != '':
            args.arch = hparams.seg_model
            args.pretrained = p
            m = main_segmentation.gen_model(args, None, data, arch=arch)
            models.append(m.cpu())
        else:
            models.append(None)
    return models


def parser():
    """
    :return: The agrsparsing arguments, arguments of main_patho_lightning.py are kept, 
             only the xai-method argument was added 
    """
    ap = main_patho_lightning.add_default_args()
    ap = main_patho_lightning.add_model_specific_args(ap)
    ap.add_argument('--xai-method', default='IntGrad', type=str)
    ap.add_argument('--xai-query', default='seg-metrics', type=str)
    ap.add_argument('--debug', action='store_true')
    ap.add_argument('--R-pooling', default='pool_avg_max', type=str)
    ap.add_argument('--y-class', default=1, type=int)
    ap.add_argument('--ig-steps', default=50, type=int)
    ap.add_argument('--nt-samples', default=5, type=int)
    ap.add_argument('--model-nuclei', type=str, default='', help='path to the statedict of the segmentation model, trained for nuclei')
    ap.add_argument('--model-tubule', type=str, default='', help='path to the statedict of the segmentation model, trained for tubule')
    ap.add_argument('--model-epithelium', type=str, default='', help='path to the statedict of the segmentation model, trained for epithelium')
    ap.add_argument('--arch-nuclei', type=str, default='unet', help='arch of the segmentation model, trained for epithelium')
    ap.add_argument('--arch-tubule', type=str, default='unet', help='arch of the segmentation model, trained for epithelium')
    ap.add_argument('--arch-epithelium', type=str, default='unet', help='arch of the segmentation model, trained for epithelium')
    ap.add_argument('--seg-size', type=int, nargs='+', default=(96, 96), help='input size for segmentation model')
    ap.add_argument('--seg-model', type=str, default='unet', help='segmentation model')
    ap.add_argument('--img-shape', nargs='+', type=int, default=(96, 96), help='input image shape')
    ap.add_argument('--plot', action='store_true', help='if specified, plots displaying segmentation and heatmap overlays will be generated')
    ap.add_argument('--max-samples', type=int, default=None, help='maximum number of samples and heatmaps to collect from the data')
    ap.add_argument('--n-head', type=int, default=-1, help='head position to inspect for Attention map')
    ap.add_argument('--all-heads', action='store_true', help='If this flag is set, all transformer heads will be visualized')
    ap.add_argument('--cls-idx', default=None, type=int, help='If this parameter is set only a certaint class will be featured in the samples')
    ap.add_argument('--cls-objective', default=None, type=int, help='If this parameter is set all explanations are for this class instead of the models output')
    return ap


def save_data(d_out, h_maps, images, masks=None):
    # helper function to generate file names: <img/hmap/mask>_<00...0><i>
    get_file_str = lambda file_name, i, N: f'{file_name}{(len(str(N))-len(str(i)))*"0"}{i}'
    # # writing images
    # img_dir = os.path.join(d_out, 'imgs')
    # if not os.path.isdir(img_dir):
    #     os.makedirs(img_dir)
    # print(f'writing images to {img_dir}')
    # for i, img in tqdm(enumerate(images), total=len(images)):
    #     fname = get_file_str('img_', i, len(images)) + '.png'
    #     cv2.imwrite(os.path.join(img_dir, fname), cv2.cvtColor((img*255).astype(np.uint8), cv2.COLOR_RGB2BGR))
    # # writing heatmaps 
    # img_dir = os.path.join(d_out, 'hmaps')
    # if not os.path.isdir(img_dir):
    #     os.makedirs(img_dir)
    # print(f'writing heatmaps to {img_dir}')
    # for i, img in tqdm(enumerate(h_maps), total=len(h_maps)):
    #     fname = get_file_str('hmap_', i, len(images)) + '.npy'
    #     np.save(os.path.join(img_dir, fname), img)
    # writing masks
    if masks is not None:
        #for k in masks.keys():
        img_dir = os.path.join(d_out, 'masks')#, k)
        if not os.path.isdir(img_dir):
            os.makedirs(img_dir)
        print(f'writing masks to {img_dir}')
        for i, img in tqdm(enumerate(masks), total=len(masks)): #tqdm(enumerate(masks[k]), total=len(masks[k])):
            fname = get_file_str('mask_', i, len(images))+'.npy'
            np.save(os.path.join(img_dir, fname), img.astype(np.uint8) * 255)
            #cv2.imwrite(os.path.join(img_dir, fname), img.astype(np.uint8) * 255)


def validate(h_maps, masks, masks_bg, masks_tissue):
    curves = [[] for _ in range(3)]
    Ks = [[] for _ in range(3)]
    diffs = [[] for _ in range(3)]
    l_pearson_rs = [[] for _ in range(3)]
    l_pearson_ps = [[] for _ in range(3)]
    weighted_rels = [[] for _ in range(3)]
    #rank_accs = [[] for _ in range(3)]
    N = 0
    N_samples = 20
    for R, F_cells, F_bg, F_tissue in tqdm(zip(h_maps, masks, masks_bg, masks_tissue), total=len(h_maps)):
        # sanity: no empty masks 
        if any([F.sum() == 0 or R.sum() == 0 for F in (F_cells, F_bg, F_tissue)]):
            #print('skipping ', i)
            continue
        F_cells *= (F_bg == 0)
        for i, F in enumerate([F_cells, F_bg, F_tissue]):
            # sanity: input size
            if not R.shape == F.shape:
                F = cv2.resize(F, R.shape, interpolation=cv2.INTER_NEAREST)
            # # integrated rank accuracy
            # if (F.sum() >= N_samples):
            #     curve, K = integrated_top_K(R, F, step_size=1/N_samples)
            #     curves[i].append(curve)
            # # difference of integrated area and relative mask/img ratio
            # diffs[i].append(curve.sum() / len(curve) - (K/np.prod(F.shape)))
            # rank accuracy with fixed K=|F|
            #rank_accs[i].append(rank_accuracy(R, F))
            # storing K-parameter
            Ks[i].append(F.sum()/np.prod(F.shape))
            if np.isnan(F).sum() > 0 or np.isnan(R).sum() > 0 or np.isinf(F).sum() > 0 or np.isinf(R).sum() > 0:
                continue
            # mass accuarcy
            weighted_rels[i].append(mass_accuracy(R, F))
            # pearson metrices
            val_pearson_r, val_pearson_p = pointbiserialr(F.flatten(), R.flatten())
            l_pearson_rs[i].append(val_pearson_r)
            l_pearson_ps[i].append(val_pearson_p)
        N += 1
    print('N', len(l_pearson_rs[0]))
    print('pearson r', [np.mean(e) for e in l_pearson_rs])
    print('pearson p', [np.mean(e) for e in l_pearson_ps])
    print('mass Acc.', [np.mean(e) for e in weighted_rels])
    #print('rank Acc.', [np.mean(e) for e in rank_accs])
    #print('curve', [np.mean(e, axis=0).mean() for e in curves])
    print('avg mask area', [np.mean(e) for e in Ks])
    #np.save(os.path.join(hparams.output_path, 'curves.npy'), np.array(curves))
    #np.save(os.path.join(hparams.output_path, 'Ks.npy'), np.array(Ks))
    #np.save(os.path.join(hparams.output_path, 'pearson_rs.npy'), np.array(l_pearson_rs))
    #np.save(os.path.join(hparams.output_path, 'mass_accs.npy'), np.array(weighted_rels))


# EXAMPLE:
# python3 main_heatmaps.py PCAM --dataset=PCAM --output-path=/media/data/out/hmaps --pretrained=/media/data/out/softmax/PCAM_ResNet_batch64_PT --refresh-rate=1 --arch=resnet50 --gpus=1 --no-concat-pooling --xai-method=IntGradNoiseTunnel --batch-size=64
if __name__ == '__main__':
    # args parsing
    hparams = parser().parse_args()

    # checking if there is dset_paths.json in key2ai-med/meta_data/, if so dataset paths can just be read from there
    hparams = main_patho_lightning.dset_shortcut(hparams)

    # automaticly setting the input size for transformers
    hparams = main_patho_lightning.transformer_input_size_sanity(hparams)

    # output sanity
    if not os.path.exists(hparams.output_path):
        os.makedirs(hparams.output_path)

    # loading the model
    model = main_patho_lightning.construct_model(hparams, requires_hooks=True)

    # loading the dataset
    data = main_patho_lightning.PathoLightning.get_dataset(hparams, split='test', trafos=transforms.ToTensor())
    _, trafos_test = main_patho_lightning.PathoLightning.prepare_transformations(hparams)
    baselines = get_data(hparams, test_transforms=model.prepare_transformations(hparams)[1])

    # when all heads are specified, all heads will be visualized, normal procedure otherwise
    vit_all_heads = hparams.all_heads and hparams.xai_method == 'Attention'
    # filter for non empty bg
    filter_bg = BackgroundSegmentation()
    if not vit_all_heads:
        # initialize the XAI method
        f_xai = get_f_xai(hparams, model)

        # generating heat maps and additional information (i.e. rank/ mass accuracy or prediction scores)
        hm_gen = HeatMapGenerator(model=model, data=data, f_xai=f_xai, batch_size=hparams.batch_size, baselines=baselines, 
                                  pool=POOL_FUNCTIOS[hparams.R_pooling],
                                  debug=hparams.debug, ig_steps=hparams.ig_steps, nt_samples=hparams.nt_samples,
                                  dir_out=hparams.output_path, trafos=transforms.Resize(main_patho_lightning.PathoLightning.image_size(hparams)),
                                  filter_fn=lambda x: True) #filter_bg(x[None,:])[:,1].sum() > 0)
        h_maps, origs, scores, gt = hm_gen.all_heatmaps(max_samples=hparams.max_samples, y_class=hparams.cls_idx, target=hparams.cls_objective)
    else:
        head_to_data = []
        output_dir = hparams.output_path
        for i in range(model.model.num_heads):
            print(f'head {i+1}/{model.model.num_heads}')
            hparams.n_head = i
            hparams.output_path = os.path.join(output_dir, f'head_{"0" * (len(str(hparams.n_heads)) - len(str(i+1)))}{i+1}')
            # initialize the XAI method
            f_xai = get_f_xai(hparams, model)

            # generating heat maps and additional information (i.e. rank/ mass accuracy or prediction scores)
            hm_gen = HeatMapGenerator(model=model, data=data, f_xai=f_xai, batch_size=hparams.batch_size, baselines=baselines, 
                                      pool=POOL_FUNCTIOS[hparams.R_pooling],
                                      debug=hparams.debug, ig_steps=hparams.ig_steps, nt_samples=hparams.nt_samples,
                                      dir_out=hparams.output_path, filter_fn=lambda x: True,
                                      trafos=transforms.Resize(main_patho_lightning.PathoLightning.image_size(hparams)))
            h_maps, origs, scores, gt = hm_gen.all_heatmaps(max_samples=hparams.max_samples)
            head_to_data.append([h_maps, origs])

    # load segmentation models
    model_cells = get_seg_models(hparams, SampleDset(samples=origs, trafo=lambda o: o.transpose(2,0,1)))[0]
    model_bg = main_segmentation.LightningSegmentation(BackgroundSegmentation(), hparams, train=None, test=SampleDset(samples=origs, trafo=lambda o: o.transpose(2,0,1)))
    model_tissue = main_segmentation.LightningSegmentation(TissueSegmentation(model_bg, model_cells), hparams, train=None, test=SampleDset(samples=origs, trafo=lambda o: o.transpose(2,0,1)))

    # apply semantic enrichment
    (masks, masks_bg, masks_tissue) = get_seg_masks(model_nuclei=model_cells, model_bg=model_bg, model_tissue=model_tissue,
                                                    max_samples=hparams.max_samples, resize=hparams.seg_size) #model.image_size(hparams))

    # calculate the metrics
    # r_top_K = float(integrated_top_K(h_maps, masks).mean())
    # r_pearson = float(pearson_biserial_r(h_maps, masks).mean())
    # print(f'validated metrices:\nr_pearson = \t{r_pearson}\nr_top_K = \t{r_top_K}')
    # # saving results
    # with open(os.path.join(hparams.output_path, 'METRICES.json'), 'w') as f:
    #     json.dump({'r_pearson': r_pearson, 'r_top_K': r_top_K}, f, indent=2)

    # # saving hparams and models
    # with open(os.path.join(hparams.output_path, 'hparams.json'), 'w') as f_hparams:
    #     json.dump(dict(vars(hparams)), f_hparams, indent=2)
    # torch.save(model.state_dict(), os.path.join(hparams.output_path, 'classifier.pth'))
    # for seg_model, name in zip(seg_models, ['nuclei', 'tubule', 'epithelium']):
    #     if seg_model is not None:
    #         torch.save(seg_model.state_dict(), os.path.join(hparams.output_path, f'segmentation_{name}.pth'))

    # clear gpu memory (Writing heatmap-plots to disk may take some time for large datasets)
    del model
    del f_xai
    del hm_gen
    #del seg_models
    torch.cuda.empty_cache()

    # saving heatmaps, original images and (if a segmentation model was specified) segmentation masks
    np.save(os.path.join(hparams.output_path, 'output_scores.npy'), scores)
    if not vit_all_heads:
        save_data(hparams.output_path, h_maps, origs, masks)
    else:
        for i, (h_maps, origs) in enumerate(head_to_data):
            save_data(os.path.join(hparams.output_path, f'head_{i}'), h_maps, origs, masks)


    # saving plots, visualizing heatmaps and segemntation mask overlays
    if hparams.plot:
        if not vit_all_heads:
            save_plots(h_maps, origs, hparams, seg_masks=masks, scores=scores, gt=gt)
        else:
            for i, (h_maps, origs) in enumerate(head_to_data):
                save_plots(h_maps, origs, hparams, seg_masks=masks, dir_suffix=f'head_{i}', scores=scores, gt=gt)

    # 
    if not vit_all_heads:
        validate(h_maps, masks, masks_bg, masks_tissue)
    else:
        for i, (h_maps, origs) in enumerate(head_to_data):
            print(f'---- HEAD {i+1} ----')
            validate(h_maps, masks, masks_bg, masks_tissue)

