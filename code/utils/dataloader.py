import numpy as np
import torch
import pandas as pd
import json
from pathlib import Path
import os
from tqdm import tqdm
from collections import defaultdict
import torchvision.transforms.functional as F_trafo
from torchvision import transforms

from torchvision.datasets.folder import default_loader
from torch.utils.data import Dataset

import elasticdeform.torch as etorch

import random

from PIL.ImageStat import Stat
from PIL import Image, ImageDraw, ImagePath

imagenet_stats = [[0.485, 0.456, 0.406],[0.229, 0.224, 0.225]]
cifar_stats = [[0.491, 0.482, 0.447],[0.247, 0.243, 0.262]]
chexpert_stats = [[0.50283729, 0.50283729, 0.50283729], [0.29132762, 0.29132762, 0.29132762]]
dr_stats = [[0.3203, 0.2244, 0.1609], [0.2622, 0.1833, 0.1318]]


label_itos_stl10 = ["airplane", "bird", "car", "cat", "deer", "dog", "horse", "monkey", "ship", "truck"]
#default label_itos for chest x-ray
label_itos_chexpert5 = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion"]
label_itos_chexpert14 = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion', 'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion', 'Lung Opacity', 'Pleural Other', 'Pneumonia', 'Pneumothorax', 'Support Devices', 'No Finding']
label_itos_cxr14 = ["Atelectasis", "Cardiomegaly", "Consolidation", "Effusion", "Edema", "Infiltration","Mass","Nodule","Pneumonia","Pneumothorax","Emphysema","Fibrosis",
"Pleural_Thickening","Hernia"]

##############################################################################
#UTILS
##############################################################################

#from fastaiv1
def _get_files(parent, p, f, extensions):
    p = Path(p)
    if isinstance(extensions,str): extensions = [extensions]
    low_extensions = [e.lower() for e in extensions] if extensions is not None else None
    res = [p/o for o in f if not o.startswith('.')
           and (extensions is None or o.split(".")[-1].lower() in low_extensions)]
    return res

def get_files(path, extensions=None, recurse=False, exclude=None,
              include=None, followlinks=False):
    "Return list of files in `path` that have a suffix in `extensions`; optionally `recurse`."
    if recurse:
        res = []
        for i,(p,d,f) in enumerate(os.walk(path, followlinks=followlinks)):
            # skip hidden dirs
            if include is not None and i==0:   d[:] = [o for o in d if o in include]
            elif exclude is not None and i==0: d[:] = [o for o in d if o not in exclude]
            else:                              d[:] = [o for o in d if not o.startswith('.')]
            res += _get_files(path, p, f, extensions)
        return res
    else:
        f = [o.name for o in os.scandir(path) if o.is_file()]
        res = _get_files(path, path, f, extensions)
        return res

##############################################################################
#GENERIC DATASET
##############################################################################        
class ImageDataframeDataset(Dataset):
    '''creates a dataset based on a given dataframe'''
    def __init__(self, df, transform=None, target_transform=None,
                     loader=default_loader, col_filename="path", col_target="label",
                     col_target_set=None, rotate_90=False, ratios=None):
        super(ImageDataframeDataset).__init__()
        self.col_target_set = col_target_set
        if ratios is not None:
            assert sum(ratios) == 1, "ratios should sum up to 1"
            # TODO
            current_ratios = []
            clss = sorted(set(np.array(df[col_target], dtype=int)))
            df_clss = []
            if isinstance(df, dict):
                df = pd.DataFrame.from_dict(df)
            for c in clss:
                is_c = np.array(df[col_target] == c)
                current_ratios.append(sum(is_c))
                df_clss.append(df.iloc[is_c])
            current_ratios = np.array(current_ratios, dtype=float)
            current_ratios /= current_ratios.sum()
            ratios = np.array(ratios)
            weights = ratios / current_ratios
            # lowest weight = 1 <=> no under-sampling (rather oversample, than undersample)
            weights /= weights.min()
            new_df = {col_filename: [], col_target: []}
            for c in clss:
                count = 0
                while count / len(df_clss[c]) < weights[c]:
                    for idx in range(len(df_clss[c])):
                        if count / len(df_clss[c]) >= weights[c]:
                            break
                        else:
                            new_df[col_filename].append(df_clss[c].iloc[idx][col_filename])
                            new_df[col_target].append(df_clss[c].iloc[idx][col_target])
                            count += 1
            df = pd.DataFrame.from_dict(new_df)
        if(col_target_set is not None):#predefined set for semi-supervised
            self.samples = list(zip(np.array(df[col_filename]), np.array(df[col_target]), np.array(df[col_target_set],dtype=np.int8)))
        else:
            self.samples = list(zip(np.array(df[col_filename]), np.array(df[col_target])))
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.labels = tuple(map(int, sorted(set(df[col_target]))))
        self.rotate_90= rotate_90
        
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        if(self.col_target_set is not None):
            path, target, subset = self.samples[index]
        else:
            path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        try:
            target = torch.tensor(target)
        except:
            target = torch.from_numpy(np.array(target))
        if self.rotate_90:
            rots = [0, 90, 180, 270]
            sample = torch.stack([transforms.functional.rotate(sample, r) for r in rots])
            target = torch.stack([target for _ in rots]).long()
        if(self.col_target_set is not None):
            return sample, [target, subset]
        else:
            return sample, target

    def collate(self, batch):
        sample, target = [], []
        for s, t in batch:
            sample.append(s)
            target.append(t)
        if self.rotate_90:
            sample = torch.cat(sample, dim=0)
            target = torch.cat(target, dim=0)
        else:
            sample = torch.stack(sample)
            target = torch.stack(target)
        return sample, target

    def __len__(self):
        return len(self.samples)



class SegmentationDataset(Dataset):
    """Creates a Segmentation dataset based on a dataframe"""

    def __init__(self, df, img_transform=transforms.ToTensor(), seg_mask_transform=transforms.ToTensor(),
                 img_loader=default_loader, seg_mask_loader=default_loader, 
                 col_img_filename='img', col_seg_mask_filename='seg-mask', augment=False, rotate_90=False):
        self.df = df
        self.img_transform = img_transform
        self.seg_mask_transform = seg_mask_transform
        self.img_loader = img_loader
        self.seg_mask_loader = seg_mask_loader
        self.col_img_filename = col_img_filename
        self.col_seg_mask_filename = col_seg_mask_filename
        self.augment = augment
        self.rotate_90 = rotate_90

    def __getitem__(self, index):
        p_img = self.df[self.col_img_filename].iloc[index]
        p_seg = self.df[self.col_seg_mask_filename].iloc[index]
        img = self.img_loader(p_img)
        seg = self.seg_mask_loader(p_seg)
        img = self.img_transform(img)
        seg = (self.seg_mask_transform(seg) > 0).type(torch.float32)
        if self.augment:
            img, seg = self.__fn_augment(img, seg)
        if self.rotate_90:
            rots = [0, 90, 180, 270]
            img = torch.stack([transforms.functional.rotate(img, r) for r in rots])
            seg = torch.stack([transforms.functional.rotate(seg, r) for r in rots])
        return img, seg

    def __fn_augment(self, t1, t2):
        """
        The internal augmentation function that is applied when self.augment is set to True.
        Transormations feature: flips, rotation, color trafos, displacement/ deformation, blurring.
        
        :param t1: the color image / input
        :param t2: the segmentation mask / target
        """
        ## random hyperparameters
        # flipping
        p_flip_h = float(torch.randn(1))
        p_flip_v = float(torch.randn(1))
        # rotation
        angle_rot = float((torch.randn(1) - 0.5) * 2 * 180) # [-180, 180]
        # hue and saturation / color
        hue_factor = float(torch.empty(1).uniform_(-0.5, 0.5))
        contrast_factor = float(torch.empty(1).uniform_(0, 2))
        saturation_factor = float(torch.empty(1).uniform_(0, 2))
        # deformation
        displacement_vector = torch.rand([2, 3, 3]) * 10
        deform = lambda E: torch.stack([etorch.deform_grid(e, displacement_vector, order=3) for e in E])
        # blurring
        kernel_size, sigma = 3, float(torch.empty(1).uniform_(0.1, 2.0).item())
        ## apply hyperparameters for transformations of both tensors
        # flipping
        if p_flip_h > 0.5:
            t1 = F_trafo.hflip(t1)
            t2 = F_trafo.hflip(t2)
        if p_flip_v > 0.5:
            t1 = F_trafo.vflip(t1)
            t2 = F_trafo.vflip(t2)
        # rotation
        t1 = F_trafo.rotate(t1, angle_rot)
        t2 = F_trafo.rotate(t2, angle_rot)
        # hue and saturation / color
        t1 = F_trafo.adjust_hue(t1, hue_factor)
        t1 = F_trafo.adjust_saturation(t1, saturation_factor)
        # deformation 
        #t1 = deform(t1)
        #t2 = deform(t2)
        # blurring
        t1 = F_trafo.gaussian_blur(t1, kernel_size, [sigma, sigma])
        return t1, t2

    def __len__(self):
        return len(self.df)

    def collate(self, batch):
        sample, target = [], []
        for s, t in batch:
            sample.append(s)
            target.append(t)
        if self.rotate_90:
            sample = torch.cat(sample, dim=0)
            target = torch.cat(target, dim=0)
        else:
            sample = torch.stack(sample)
            target = torch.stack(target)
        return sample, target

    def stretch(self, factor):
        """
        If there are only ~1k images, the number of epochs and therefore the number of pytorchlightning evaluations and 
        checkpoints grows rather large before convergence is met.
        Therefore stretching the dataset, i.e. repeating all samples, can be usefull.
        """
        self.df = pd.concat([self.df for _ in range(factor)])


################################################################################
# Histo Datasets
################################################################################


class RobustnessSampleSet(ImageDataframeDataset):

    def __init__(self, base_dir, split='test', *args, **kwargs):
        pths = self.__get_files(base_dir)
        lbls = self.__extract_labels(pths)
        df = {}
        df['path'] = pths
        df['label'] = lbls
        df = pd.DataFrame.from_dict(df)
        super().__init__(df=df, *args, **kwargs)
        self.__norm_stats = None

    def __get_files(self, root, sfx=['.png', '.jpg', '.jpeg', '.tiff']):
        print(root)
        pths = []
        for r, _, fs in os.walk(root):
            for f in fs:
                if any([f.lower().endswith(s) for s in sfx]):
                    pths.append(os.path.join(r, f))
        return pths
    
    def __extract_labels(self, pths):
        lbls = []
        for p in pths:
            lbl = int(p.split('[')[1].split(']')[0])
            lbls.append(lbl)
        return lbls

    @property
    def NORM_STATS(self):
        if self.__norm_stats is None:
            self.__norm_stats = self.__calc_norm_stats()
        return self.__norm_stats

    def __calc_norm_stats(self, verbose=True):
        mean = torch.zeros(3)
        if verbose:
            print('Calculating mean colour channels.')
            pbar = tqdm(total=len(self))
        for img, _ in self:
            mean += img.mean(dim=1).mean(dim=1)
            if verbose:
                pbar.update(1)
        mean /= len(self)
        std = torch.zeros(3)
        if verbose:
            print('Calculating std for the colour channels.')
            pbar = tqdm(total=len(self))
        for img, _ in self:
            std += torch.sqrt(img.mean(dim=1).mean(dim=1) ** 2)
            if verbose:
                pbar.update(1)
        std /= len(self)
        # keeping stats as list of list for type consistency between datasets
        stats = [mean.detach().numpy().tolist(), std.detach().numpy().tolist()]
        return stats


class PCam(ImageDataframeDataset):
    NORM_STATS = [[0.6908793, 0.5314163, 0.69249374], [0.2124685, 0.27969167, 0.23778248]]

    def __init__(self, base_dir, split='train', transform=None, target_transform=None,
                 col_filename="path", col_target="label", col_target_set=None, loader=default_loader, **kwargs):
        df = pd.read_csv(os.path.join(base_dir, 'annotations.csv'))
        df_split = df[[split in line for line in df['path']]]
        new_frame = {'path': None, 'label': None}
        new_frame['label'] = df_split['label']
        new_frame['path'] = [os.path.join(base_dir, r) for r in df_split['path']]
        df_split = pd.DataFrame.from_dict(new_frame)
        # you can either specify the split or extract the splits df after initialization
        self.df = df_split
        super().__init__(df=df_split, transform=transform, target_transform=target_transform, col_filename=col_filename,
                         col_target=col_target, col_target_set=col_target_set, loader=loader, **kwargs)

class LungColon(ImageDataframeDataset):
    NORM_STATS = None

    def __init__(self, base_dir, split='train', transform=None, target_transform=None,
                 col_filename="path", col_target="label", col_target_set=None, loader=default_loader, **kwargs):
        imgs, lbls = self.__search(base_dir)
        N = len(imgs)
        idcs = np.random.RandomState(seed=42).permutation(N)
        df = {col_filename: imgs[idcs], col_target: lbls[idcs]}
        f_split = lambda dct, pct_s, pct_e: {k: dct[k][int(len(dct[k])*pct_s) : int(len(dct[k])*pct_e)] for k in dct.keys()}
        splits = {
            'train': f_split(df, 0, 0.75), 
            'valid': f_split(df, 0.75, 0.875), 
            'test': f_split(df, 0.875, 1.0), 
        }
        df_split = splits[split]
        self.df = df_split
        super().__init__(df=df_split, transform=transform, target_transform=target_transform, col_filename=col_filename,
                         col_target=col_target, col_target_set=col_target_set, loader=loader, **kwargs)


    def __search(self, base_dir, suffixes=['.jpeg', '.jpg', '.png']):
        LUNG_DIR = 'lung_image_sets'
        COLON_DIR = 'colon_image_sets'
        lung_imgs = os.path.join(base_dir, LUNG_DIR) 
        colon_imgs = os.path.join(base_dir, COLON_DIR) 
        # search lung data
        lung_paths = []
        lung_classes = []
        for r, _, fs in os.walk(lung_imgs):
            for f in fs:
                if any([f.endswith(s) for s in suffixes]):
                    f_path = os.path.join(r, f)
                    dir_name = os.path.dirname(f_path)
                    cls = dir_name.split('_')[-1]
                    lung_paths.append(f_path)
                    lung_classes.append(cls)
        C_lung = set(lung_classes)
        C_lung_mapping = {k: i for i, k in enumerate(C_lung)}
        lung_classes = [C_lung_mapping[c] for c in lung_classes]
        # search colon data
        colon_paths = []
        colon_classes = []
        for r, _, fs in os.walk(colon_imgs):
            for f in fs:
                if any([f.endswith(s) for s in suffixes]):
                    f_path = os.path.join(r, f)
                    dir_name = os.path.dirname(f_path)
                    cls = dir_name.split('_')[-1]
                    colon_paths.append(f_path)
                    colon_classes.append(cls)
        C_colon = set(lung_classes)
        C_colon_mapping = {k: len(C_lung) + i for i, k in enumerate(C_lung)}
        colon_classes = [C_colon_mapping[c] for c in colon_classes]

        return np.array(lung_paths + colon_paths), np.array(lung_classes + colon_classes)


class IDCBreastHisto(ImageDataframeDataset):
    NORM_STATS = [[0.8118834, 0.6294317, 0.73130006], [0.14308669, 0.21005107, 0.15084313]]

    def __init__(self, base_dir, split='train', transform=None, target_transform=None,
                 col_filename="path", col_target="label", col_target_set=None, loader=default_loader, **kwargs):
        paths = self.__search_imgs(base_dir)
        lbls = [int(p.split('class')[1][0]) for p in paths]
        paths, lbls = map(np.array, (paths, lbls))
        split_to_file = {
            'test': os.path.join(base_dir, 'cases_test.txt'),
            'train': os.path.join(base_dir, 'cases_train.txt'),
            'valid': os.path.join(base_dir, 'cases_val.txt')
        }
        if not all([os.path.isfile(split_to_file[s]) for s in ['test', 'train', 'valid']]):
            print('WARNING: not all fold files are in located in {base_dir}, resorting to random splits.')
            N = len(paths)
            idcs = np.random.RandomState(seed=42).permutation(N)
            if split == 'train':
                b, e = 0, int(0.75 * N)
            elif split == 'valid':
                b, e = int(0.75 * N), int(0.875 * N)
            elif split == 'test':
                b, e = int(0.875 * N), N
            paths_split = paths[b:e]
            lbls_split = lbls[b:e]
        else:
            with open(split_to_file[split], 'r') as f:
                wsis = [wsi.strip() for wsi in f.readlines()]
            idcs = []
            paths_split = []
            lbls_split = []
            for i in range(len(paths)):
                for wsi in wsis:
                    if wsi in paths[i]:
                        paths_split.append(paths[i])
                        lbls_split.append(lbls[i])
            paths_split = np.array(paths_split)
            lbls_split = np.array(lbls_split)
        df_split = pd.DataFrame.from_dict({col_filename: paths_split, col_target: lbls_split})
        super().__init__(df=df_split, transform=transform, target_transform=target_transform, col_filename=col_filename,
                         col_target=col_target, col_target_set=col_target_set, loader=loader, **kwargs)

    def __search_imgs(self, root, suffixes=['.png', '.jpg', '.jpeg', '.tiff']):
        pths = []
        for r, _, fs in os.walk(root):
            for f in fs:
                if any([f.lower().endswith(s) for s in suffixes]):
                    pths.append(os.path.join(r, f))
        return pths


class MHIST(ImageDataframeDataset):
    NORM_STATS = [[0.7404394, 0.65518796, 0.7787634], [0.19584168, 0.2416044, 0.16876698]]

    def __init__(self, base_dir, split='train', transform=None, target_transform=None,
                 col_filename="path", col_target="label", col_target_set=None, loader=default_loader, **kwargs):
        assert split in ['train', 'test'], f"Only train and test split are available, the split: {split} is not recognized"
        df = pd.read_csv(os.path.join(base_dir, 'annotations.csv'))
        new_frame = {'path': None, 'label': None}
        new_frame['path'] = [os.path.join(base_dir, 'images', p) for p in df['Image Name']]
        new_frame['label'] = [0 if l == 'HP' else 1 for l in df['Majority Vote Label']]
        df_split = pd.DataFrame.from_dict(new_frame)[df['Partition']==split]
        self.df = df_split
        super().__init__(df=df_split, transform=transform, target_transform=target_transform, col_filename=col_filename,
                         col_target=col_target, col_target_set=col_target_set, loader=loader, **kwargs)


class BreaKHis(ImageDataframeDataset):
    NORM_STATS = [[0.7879071, 0.6226316, 0.7679345], [0.11364735, 0.16602136, 0.09562508]]

    def __init__(self, base_dir, split='train', transform=None, target_transform=None, loader=default_loader, magn=40,
                 multiclass=True, use_stored_fold=True, **kwargs):
        print(f'initializing x{magn} magnification subset of BreaKHis')
        if not base_dir[-1] == '/':
            base_dir += '/'
        # walk dataset subdirectories
        #bening_imgs         = np.array([e.split(base_dir)[1] for e in self.__search(os.path.join(base_dir, 'benign'), magn=magn)])
        # adenosis  fibroadenoma  phyllodes_tumor  tubular_adenoma
        benign_adenosis         = np.array([e.split(base_dir)[1] for e in self.__search(os.path.join(base_dir, 'benign', 'SOB', 'adenosis'), magn=magn)])
        benign_fibroadenoma     = np.array([e.split(base_dir)[1] for e in self.__search(os.path.join(base_dir, 'benign', 'SOB', 'fibroadenoma'), magn=magn)])
        benign_phyllodes_tumor  = np.array([e.split(base_dir)[1] for e in self.__search(os.path.join(base_dir, 'benign', 'SOB', 'phyllodes_tumor'), magn=magn)])
        benign_tubular_adenoma  = np.array([e.split(base_dir)[1] for e in self.__search(os.path.join(base_dir, 'benign', 'SOB', 'tubular_adenoma'), magn=magn)])

        ductal_carcinoma    = np.array([e.split(base_dir)[1] for e in self.__search(os.path.join(base_dir, 'malignant', 'SOB', 'ductal_carcinoma'), magn=magn)])
        lobular_carcinoma   = np.array([e.split(base_dir)[1] for e in self.__search(os.path.join(base_dir, 'malignant', 'SOB', 'lobular_carcinoma'), magn=magn)])
        mucinous_carcinoma  = np.array([e.split(base_dir)[1] for e in self.__search(os.path.join(base_dir, 'malignant', 'SOB', 'mucinous_carcinoma'), magn=magn)])
        papillary_carcinoma = np.array([e.split(base_dir)[1] for e in self.__search(os.path.join(base_dir, 'malignant', 'SOB', 'papillary_carcinoma'), magn=magn)])
        # setup the split
        data_distribution = {'train': (0,0.5), 'valid': (0.5, 0.75), 'test': (0.75, 1.0)}
        fold_loc = os.path.join(base_dir, f'BreaKHis_x{magn}_folds.json')
        if use_stored_fold and os.path.isfile(fold_loc):
            print(f'using split meta-data at {fold_loc}')
            with open(fold_loc, 'r') as f:
                splits = json.load(f)
        else:
            # setting seed => same split every time (TODO: cross validation??)
            np.random.seed(42)
            idcs_benign_ad, idcs_benign_fi, idcs_benign_ph, idcs_benign_tu, \
            idcs_du, idcs_lo, idcs_mu, idcs_pa = map(
                lambda l: np.random.permutation(len(l)),
                [benign_adenosis, benign_fibroadenoma, benign_phyllodes_tumor, benign_tubular_adenoma,
                 ductal_carcinoma, lobular_carcinoma, mucinous_carcinoma, papillary_carcinoma]
            )
            splits_idcs = np.concatenate([idcs_benign_ad, idcs_benign_fi, idcs_benign_ph, idcs_benign_tu,
                                          idcs_du, idcs_lo, idcs_mu, idcs_pa])
            splits = {}
            for key in data_distribution.keys():
                pct_range = data_distribution[key]
                split_benign_ad, split_benign_fi, split_benign_ph, split_benign_tu,\
                split_du, split_lo, split_mu, split_pa = [
                    d[ i[int(pct_range[0]*len(d)) : int(pct_range[1]*len(d))] ].tolist()
                    for d, i in zip([benign_adenosis, benign_fibroadenoma, benign_phyllodes_tumor, benign_tubular_adenoma,
                                     ductal_carcinoma, lobular_carcinoma, mucinous_carcinoma, papillary_carcinoma],
                                    [idcs_benign_ad, idcs_benign_fi, idcs_benign_ph, idcs_benign_tu,
                                     idcs_du, idcs_lo, idcs_mu, idcs_pa])
                ]
                split_paths = [split_benign_ad, split_benign_fi, split_benign_ph, split_benign_tu,
                               split_du, split_lo, split_mu, split_pa]
                splits[key] = split_paths
            with open(fold_loc, 'w') as f:
                json.dump(splits, f)

        # setup the dataframe
        paths =  np.array(splits[split])
        [split_benign_ad, split_benign_fi, split_benign_ph, split_benign_tu, split_du, split_lo, split_mu, split_pa] = paths
        if multiclass:
            labels = np.concatenate([np.ones(len(d)) * i 
                                     for i, d in enumerate([split_benign_ad, split_benign_fi, split_benign_ph, split_benign_tu, 
                                                            split_du, split_lo, split_mu, split_pa])])
        else:
            labels = np.concatenate([np.ones(len(d)) * int(i==4) 
                                     for i, d in enumerate([split_benign_ad, split_benign_fi, split_benign_ph, split_benign_tu, 
                                                            split_du, split_lo, split_mu, split_pa])])
        df_split = {'path': np.core.defchararray.add(base_dir, np.concatenate(paths)), 'label': labels}
        self.df = pd.DataFrame.from_dict(df_split)
        # ImageDataframeDataset initialization
        super().__init__(df=df_split, transform=transform, target_transform=target_transform,
                         col_filename='path', col_target='label', col_target_set=None, loader=loader, **kwargs)

    def __search(self, base_dir, suffix='.png', magn=200):
        l = []
        for root, _, files in os.walk(base_dir):
            for f in files:
                if f.endswith(suffix):
                    if int(f.split('-')[-2]) == magn:
                        fname = os.path.join(root, f)
                        l.append(fname)
        return np.array(sorted(l))


class GasHisSDB(ImageDataframeDataset):
    NOMR_STATS = [[0.90394706, 0.7863083, 0.9073586], [0.12587382, 0.17743136, 0.09878214]]

    def __init__(self, root_dir, split='train', pixels=160, transform=None, target_transform=None, loader=default_loader,
                 **kwargs):
        assert pixels in [80, 120, 160], f'Image sizes of {[f"{i}x{j}" for i, j in zip([80, 120, 160], [80, 120, 160])]} are available in GasHisSDB, {pixels}x{pixels} is not.'
        self.pixels = pixels
        # gather images
        normal, abnormal = self.__search(os.path.join(root_dir, f'{self.pixels}'))
        idcs_normal, idcs_abnormal = map(lambda l: [int(len(l)*0.4), int(len(l)*0.8)], [normal, abnormal])
        # splits
        if split == 'train':
            normal, abnormal = normal[:idcs_normal[0]], abnormal[:idcs_abnormal[0]]
        elif split == 'valid':
            normal, abnormal = normal[idcs_normal[0]:idcs_normal[1]], abnormal[idcs_abnormal[0]:idcs_abnormal[1]]
        elif split == 'test':
            normal, abnormal = normal[idcs_normal[1]:], abnormal[idcs_abnormal[1]:]
        else:
            raise ValueError(f'Only test and train splits for {type(self)}')
        lbls_normal, lbls_abnormal = [0 for _ in normal], [1 for _ in abnormal]
        # dataframe
        paths = np.concatenate([normal, abnormal])
        labels = np.concatenate([lbls_normal, lbls_abnormal])
        df_split = {'path': paths, 'label': labels}
        self.df = pd.DataFrame.from_dict(df_split)
        super().__init__(df=df_split, transform=transform, target_transform=target_transform,
                         col_filename='path', col_target='label', col_target_set=None, loader=loader, **kwargs)

    def __search(self, pth, suffix='.png'):
        normal = sorted([os.path.join(pth, 'Normal', f) for f in os.listdir(os.path.join(pth, 'Normal')) if f.endswith(suffix)])
        abnormal = sorted([os.path.join(pth, 'Abnormal', f) for f in os.listdir(os.path.join(pth, 'Abnormal')) if f.endswith(suffix)])
        return normal, abnormal


class MIDOG(ImageDataframeDataset):
    NOMR_STATS = None

    def __init__(self, root_dir, scanner_id=1, split='all', pixels=160, transform=None, target_transform=None, loader=default_loader):
        # sanity of scanner selection
        assert split == 'all', 'currently there is no train/ test split support. For the moment the dataset can only be taken as a whole.'
        scanner_ids = np.arange(1,5)
        self.scanner_ids_str = ['Hamamatsu XR', 'Hamamatsu S360', 'Aperio ScanScope CS2', 'Leica GT450']
        self.img_ids = [(1,50), (51, 100), (101, 150), (151, 200)]
        if isinstance(scanner_id, str):
            assert scanner_id in self.scanner_ids_str, f'If you specify a scanner-name, choose from: {self.scanner_ids_str}'
            idx = scanner_ids_str.index(scanner_id)
        else:
            assert scanner_id in scanner_ids, f'If you specify a scanner-ID, choose from: {scanner_ids}'
            idx = scanner_id - 1
        # checking for patches
        self.__dset_sanity(idx, root_dir, loader=loader)
        paths = self.__search(os.path.join(root_dir, self.scanner_ids_str[idx]))
        # initialize dataset
        df_split = {'path': paths, 'label': np.zeros(len(paths))}
        super().__init__(df=df_split, transform=transform, target_transform=target_transform,
                         col_filename='path', col_target='label', col_target_set=None, loader=loader)

    def __dset_sanity(self, idx, root_dir, loader=default_loader, suffix='.png'):
        if not self.scanner_ids_str[idx] in os.listdir(root_dir):
            os.makedirs(os.path.join(root_dir, self.scanner_ids_str[idx]))
            with open(os.path.join(root_dir, 'MIDOG.json'), 'r') as f:
                annotations = json.load(f)['annotations']
            # filter annotations for images with desired scanner
            id_min, id_max = self.img_ids[idx]
            annotations = [a for a in annotations if id_min <= a['image_id'] <= id_max]
            # sorting by image_id reduces the I/O operations to a constant O(50) insted of O(N_ids)
            annotations = sorted(annotations, key=lambda a: a['image_id'])
            current_id = -1
            print(f'Patches have not been extracted yet. Generating data in {os.path.join(root_dir, self.scanner_ids_str[idx])} ...')
            for a in tqdm(annotations, total=len(annotations)):
                # loading new image if needed
                if a['image_id'] != current_id:
                    current_id = a['image_id']
                    leading_0 = (3 - len(str(current_id))) * '0'
                    img_pth = os.path.join(root_dir, f'{leading_0}{current_id}.tiff')
                    img = loader(img_pth)
                # generating patch
                r0, c0, r1, c1 = a['bbox']
                patch = img.crop((c0,r0,c1,r1))
                # writing patch to disk
                patch.save(os.path.join(root_dir, self.scanner_ids_str[idx], f'{a["id"]}{suffix}'))

    def __search(self, pth, suffix='.png'):
        return [os.path.join(pth, f) for f in os.listdir(pth) if f.endswith(suffix)]


class JanowczykSeg(SegmentationDataset):

    def __init__(self, root, split=None, img_transform=lambda x: x, seg_mask_transform=lambda x: x, 
                 img_loader=default_loader, seg_mask_loader=default_loader, 
                 col_img_filename='img', col_seg_mask_filename='seg-mask', augment=False, **kwargs):
        img_paths, seg_paths = map(np.array, self._search(root))
        if not split is None:
            np.random.seed(42)
            idcs = np.random.permutation(len(img_paths)) 
            splits = {
                'train': idcs[:int(0.75 * len(img_paths))], 
                'test': idcs[int(0.75 * len(img_paths)):]
                #'test': idcs[int(0.75 * len(img_paths)):int(0.875 * len(img_paths))],
                #'valid': idcs[int(0.875 * len(img_paths)):]
            }
            img_paths = img_paths[splits[split]]
            seg_paths = seg_paths[splits[split]]
        df = pd.DataFrame.from_dict({col_img_filename: img_paths, col_seg_mask_filename: seg_paths})
        super().__init__(df=df, img_transform=img_transform, seg_mask_transform=seg_mask_transform,
                         img_loader=img_loader, seg_mask_loader=seg_mask_loader,
                         col_img_filename=col_img_filename, col_seg_mask_filename=col_seg_mask_filename, augment=augment,
                         **kwargs)

    def _search(self, root):
        raise NotImplementedError('please implement the search function for this dataset')

   
class JanowczykNuclei(JanowczykSeg):
    NORM_STATS = [[0.6915139, 0.50203073, 0.65178776], [0.17053954, 0.18893757, 0.14069661]]

    def __init__(self, root, split=None, img_transform=transforms.ToTensor(), seg_mask_transform=transforms.ToTensor(),
                 img_loader=default_loader, seg_mask_loader=default_loader, 
                 col_img_filename='img', col_seg_mask_filename='seg-mask', augment=False, **kwargs):
        super().__init__(root, split=split, img_transform=img_transform, seg_mask_transform=seg_mask_transform, 
                         img_loader=img_loader, seg_mask_loader=seg_mask_loader, 
                         col_img_filename=col_img_filename, col_seg_mask_filename=col_seg_mask_filename, augment=augment,
                         **kwargs)

    def _search(self, root, suffixes=['.png', '.tif', 'jpg']):
        img_paths, seg_paths = [], []
        for r, _, fs in os.walk(root):
            for f in fs:
                if any([f.endswith(sfx) for sfx in suffixes]):
                    # check whether its an image or seg-mask
                    if 'original' in f:
                        img_paths.append(os.path.join(r, f))
                    elif 'mask' in f:
                        seg_paths.append(os.path.join(r, f))
                    else:
                        print(f"{os.path.join(r, f)} was not specified propperly, hence it's beeing skipped.")
        # sorting the lists of strings ensures element-wise affiliation
        return sorted(img_paths), sorted(seg_paths)


class JanowczykNucleiClassification(JanowczykNuclei):

    def __init__(self, base_dir, split='train', transform=transforms.ToTensor(), loader=default_loader, stretch=1,
                 balance=True, **kwargs):
        super().__init__(base_dir, split=split, img_transform=transform, img_loader=loader, augment=False, **kwargs)
        self.stretch(stretch)
        self.balance = False
        self.label_lookup = torch.zeros(len(self), dtype=torch.long)
        self.label_calculated = torch.zeros(len(self), dtype=bool)
        self.balance = balance
        lbls = []
        print('preparing labels')
        for i in tqdm(range(len(self.df))):
            _, lbl = self[i]
            lbls.append(lbl)
        lbls = np.array(lbls)
        self.idcs_neg = lbls == 0
        self.idcs_pos = lbls > 0
        self.offset_len = abs(np.sum(self.idcs_pos) - np.sum(self.idcs_neg))
        self.labels = (0, 1)

    def __getitem__(self, idx):
        # balancing
        if idx >= super().__len__():
            if not self.balance:
                raise IndexError()
            if self.idcs_neg.sum() > self.idcs_pos.sum():
                idcs = self.idcs_pos
            elif self.idcs_neg.sum() < self.idcs_pos.sum():
                idcs = self.idcs_neg
            else:
                raise IndexError()
            idcs = np.arange(len(idcs)) * idcs
            idcs = idcs[idcs>0]
            idx = np.random.choice(idcs)
        # data collection
        p_img = self.df[self.col_img_filename].iloc[idx]
        img = self.img_loader(p_img)
        img = self.img_transform(img)
        if self.label_calculated[idx]:
            lbl = self.label_lookup[idx]
        else:
            p_seg = self.df[self.col_seg_mask_filename].iloc[idx]
            seg = self.seg_mask_loader(p_seg)
            seg = (self.seg_mask_transform(seg) > 0).type(torch.float32)
            lbl = (seg.sum()>0).type(torch.LongTensor) 
            self.label_lookup[idx] = lbl
            self.label_calculated[idx] = True
            if self.label_calculated.sum() == len(self.label_calculated):
                self.seg_mask_loader = lambda x: torch.zeros([1])
        return img, lbl

    def __len__(self):
        N = super().__len__()
        if self.balance:
            N += self.offset_len
        return N

    def get_mask(self, idx):
        p_seg = self.df[self.col_seg_mask_filename].iloc[idx]
        seg = self.seg_mask_loader(p_seg)
        seg = (torch.mean(self.seg_mask_transform(seg), dim=0) > 0).type(torch.float32)
        return seg


class JanowczykEpi(JanowczykSeg):
    NORM_STATS = [[0.70540035, 0.48989287, 0.64631426], [0.1692513, 0.18466637, 0.13559514]]

    def __init__(self, root, split=None, img_transform=lambda x: x, seg_mask_transform=lambda x: x, 
                 img_loader=default_loader, seg_mask_loader=default_loader, 
                 col_img_filename='img', col_seg_mask_filename='seg-mask', augment=False):
        super().__init__(root, split=split, img_transform=img_transform, seg_mask_transform=seg_mask_transform, 
                         img_loader=img_loader, seg_mask_loader=seg_mask_loader, 
                         col_img_filename=col_img_filename, col_seg_mask_filename=col_seg_mask_filename, augment=augment)

    def _search(self, root, suffixes=['.png', '.tif', 'jpg']):
        has_both = defaultdict(list)
        for r, _, fs in os.walk(root):
            for f in fs:
                if any([f.endswith(sfx) for sfx in suffixes]):
                    pth = os.path.join(r, f)
                    # check whether its an image or seg-mask
                    if 'mask' in f:
                        has_both[pth.split('mask')[0]].append(pth)
                    else:
                        has_both[pth.split('patch')[0]].append(pth)
        img_paths, seg_paths = [], []
        suffix_img, suffix_mask = '', ''
        for k in has_both:
            msks = [p for p in has_both[k] if 'mask' in p]
            imgs = [p for p in has_both[k] if not 'mask' in p]
            if len(msks) == len(imgs):
                seg_paths += msks
                img_paths += imgs
        # sorting the lists of strings ensures element-wise affiliation
        return sorted(img_paths), sorted(seg_paths)


class JanowczykTubule(JanowczykSeg):
    NORM_STATS = [[0.78886235, 0.5125227, 0.786121], [0.15695812, 0.2383755, 0.12309671]]

    def __init__(self, root, split=None, img_transform=lambda x: x, seg_mask_transform=lambda x: x, 
                 img_loader=default_loader, seg_mask_loader=default_loader, 
                 col_img_filename='img', col_seg_mask_filename='seg-mask', augment=False):
        super().__init__(root, split=split, img_transform=img_transform, seg_mask_transform=seg_mask_transform, 
                         img_loader=img_loader, seg_mask_loader=seg_mask_loader, 
                         col_img_filename=col_img_filename, col_seg_mask_filename=col_seg_mask_filename, augment=augment)

    def _search(self, root, suffixes=['.png', '.tif', 'jpg']):
        img_paths, seg_paths = [], []
        for r, _, fs in os.walk(root):
            for f in fs:
                if any([f.endswith(sfx) for sfx in suffixes]):
                    # check whether its an image or seg-mask
                    pth = os.path.join(r, f)
                    if 'anno' in f:
                        seg_paths.append(pth)
                    else:
                        img_paths.append(pth)
        # sorting the lists of strings ensures element-wise affiliation
        return sorted(img_paths), sorted(seg_paths)


class DataScienceBowlNuclei(SegmentationDataset):
    NORM_STATS = [[0.19118652, 0.17835087, 0.19697899], [0.28015086, 0.26546115, 0.28681672]]

    def __init__(self, root, split=None, **kwargs):
        df = DataScienceBowlNuclei.get_split(root, split)
        super().__init__(df, seg_mask_loader=DataScienceBowlNuclei._seg_mask_loader, **kwargs)

    @staticmethod
    def _seg_mask_loader(data):
        coords, shape = json.loads(data)
        # fill in all nuclei polygons
        img = DataScienceBowlNuclei.enc_to_arr(np.concatenate(coords), shape)
        # convert to PIL Image for torchvision compatibility
        return Image.fromarray((255 * (img>0)).astype(np.uint8))

    @staticmethod
    def get_split(root, split):
        # select split
        csv_select = {
            'test': os.path.join(root, 'stage1_solution.csv'),
            'train': os.path.join(root, 'stage1_train_labels.csv')
        }
        # load raw representation
        df = pd.read_csv(csv_select[split], delimiter=',')
        # load lists of pixel encodings
        coords_dict = defaultdict(list)
        hw = {}
        for i in range(len(df)):
            if len(df.iloc[i]) >= 4:
                imageId, outline, h, w = df.iloc[i][:4]
            else: 
                imageId, outline = df.iloc[i][:2]
                if not imageId in hw.keys():
                    w, h = default_loader(os.path.join(root, imageId, 'images', f'{imageId}.png')).size
                else:
                    h, w = hw[imageId]
            outline = [int(int_str) for int_str in outline.split(' ')]
            coords_dict[imageId].append(outline)
            hw[imageId] = (int(h), int(w))
        # convert lists to JSON strings, so they can be stored in a DataFrame
        data = {}
        for k in coords_dict.keys():
            data[k] = json.dumps([coords_dict[k], hw[k]])
        # generate DataFrame: img holds paths, seg-mask holds JSON encodings of nuclei and Image shape
        imgIds = list(data.keys())
        df = {'img': [os.path.join(root, k, 'images', f'{k}.png') for k in imgIds], 'seg-mask': [data[k] for k in imgIds]}
        df = pd.DataFrame.from_dict(df)
        return df

    @staticmethod
    def enc_to_arr(coords, shape):
        img = np.zeros(shape[0]*shape[1])
        skips, steps = coords[0::2], coords[1::2]
        idcs = np.concatenate([np.arange(s, s+k)-1 for s, k in zip(skips, steps)])
        img[idcs] = 1
        return img.reshape(shape, order='F')


class PatchedMoNuSeg(SegmentationDataset):
    NORM_STATS = [[0.6200343, 0.44794205, 0.6327263], [0.17998989, 0.23317572, 0.21338217]]

    def __init__(self, root, split, **kwargs):
        df = PatchedMoNuSeg.gen_df(root, split)
        super().__init__(df, **kwargs)

    @staticmethod
    def gen_df(root, split, suffix='.tif'):
        split_to_dir = {
            'train': 'MoNuSegTrainingData',
            'test': 'MoNuSegTestData'
        }
        d_files = os.path.join(root, split_to_dir[split])
        imgs = []
        masks = []
        for f in sorted(os.listdir(d_files)):
            abs_pth = os.path.join(d_files, f)
            if not os.path.isfile(abs_pth) and f.endswith(suffix):
                continue
            if 'mask' in f:
                masks.append(abs_pth)
            else:
                imgs.append(abs_pth)
        df = pd.DataFrame.from_dict({'img': imgs, 'seg-mask': masks})
        return df


def prepare_MHIST_df(base_dir):
    split_names = 'train', 'test'
    return df_from_Dataset(base_dir=base_dir, split_names=split_names, constructor=MHIST)


def prepare_PCam_df(base_dir):
    split_names = 'train', 'test', 'valid'
    return df_from_Dataset(base_dir=base_dir, split_names=split_names, constructor=PCam)


def prepare_BreaKHis_df(base_dir):
    split_names = 'train', 'test', 'valid'
    return df_from_Dataset(base_dir=base_dir, split_names=split_names, constructor=BreaKHis)


def df_from_Dataset(base_dir, split_names, constructor):
    # fetching the splits data-frames
    df_splits = list(map(lambda x: x.df, [constructor(base_dir, split=s) for s in split_names]))
    # fetching different labels
    label_itos = np.unique(np.concatenate([df['label'] for df in df_splits]))
    ret = *df_splits, label_itos
    return ret


class BinaryEmb:

    def __init__(self):
        pass

    def __call__(self, lbl):
        emb = torch.ones([1], dtype=torch.float32) * lbl
        return emb

    def __repr__(self):
        format_string = self.__class__.__name__
        return format_string


class OneHotEnc:

    def __init__(self, n_classes):
        self.n_classes = n_classes

    def __call__(self, lbl):
        emb = torch.zeros([self.n_classes], dtype=torch.float32)
        emb[lbl] = 1
        return emb

    def __repr__(self):
        format_string = self.__class__.__name__
        return format_string
    

##############################################################################
#PREPARING DATAFRAMES
##############################################################################
def prepare_imagefolder_df(root,label_itos=None,extensions=["jpg","jpeg","png"]):
    '''prepares dataframe for imagefolder dataset'''
    files = get_files(root,extensions=extensions,recurse=True)
    df=pd.DataFrame(files,columns=["path"])
    df["label_raw"]=df.path.apply(lambda x: x.parts[-2])
    if(label_itos is None):
        label_itos = np.unique(df["label_raw"])
    label_stoi = {s:i for i,s in enumerate(label_itos)}
    df["label"] = df.label_raw.apply(lambda x:label_stoi[x])
    print(root,":",len(df),"samples.")
    return df,label_itos

##############################################################################
#CHEST X-RAY
##############################################################################

def chexpert_label_raw_to_label_uoneslsr(label_raw):
    '''stochastic label transformation from 1911.06475'''
    idxs = np.where(label_raw==-1.)[0]
    for idx in idxs:
        label_raw[idx]=random.uniform(0.55,0.85)
    return label_raw

def chexpert_label_raw_to_label_uzeroslsr(label_raw):
    '''stochastic label transformation from 1911.06475'''
    idxs = np.where(label_raw==-1.)[0]
    for idx in idxs:
        label_raw[idx]=random.uniform(0.0,0.3)
    return label_raw

def prepare_chexpert_df(root, label_itos = label_itos_chexpert5,chexphoto=False,select_frontal_lateral_train=None,select_frontal_lateral_test=None,full_dataset_as_test=False):
    '''prepares dataframe for the chexpert dataset'''
    root = Path(root)/"CheXpert-v1.0-small" if chexphoto is False else Path(root)/"CheXphoto-v1.0-small"
    df_train=pd.read_csv(root/"train.csv")
    df_test=pd.read_csv(root/"valid.csv")
    
    label_map_uones={0.0:0, 1.0:1, -1.0:1}
    label_map_uzeros={0.0:0, 1.0:1, -1.0:0}
    label_map_raw={0.0:0, 1.0:1, -1.0:-1}

    def apply_map(x,mp):
        if(np.isnan(x)):#set unmentioned to 0
            return 0.
        else:
            return mp[x]
        
    #multi-hot encoded labels
    df_train["label_uones"]=df_train.apply(lambda row: np.array([apply_map(row[c],label_map_uones) for c in label_itos]).astype(np.float32),axis=1)
    df_test["label_uones"]=df_test.apply(lambda row: np.array([apply_map(row[c],label_map_uones) for c in label_itos]).astype(np.float32),axis=1)
    df_train["label_uzeros"]=df_train.apply(lambda row: np.array([apply_map(row[c],label_map_uzeros) for c in label_itos]).astype(np.float32),axis=1)
    df_test["label_uzeros"]=df_test.apply(lambda row: np.array([apply_map(row[c],label_map_uzeros) for c in label_itos]).astype(np.float32),axis=1)
    df_train["label_raw"]=df_train.apply(lambda row: np.array([apply_map(row[c],label_map_raw) for c in label_itos]).astype(np.float32),axis=1)
    df_test["label_raw"]=df_test.apply(lambda row: np.array([apply_map(row[c],label_map_raw) for c in label_itos]).astype(np.float32),axis=1)
    df_train["label"]=df_train["label_uones"] #default label
    df_test["label"]=df_test["label_uones"] #default label
    #full paths
    df_train["path"]=df_train.Path.apply(lambda x:root/Path(x).relative_to("./CheXpert-v1.0-small" if chexphoto is False else "./CheXphoto-v1.0"))
    df_test["path"]=df_test.Path.apply(lambda x:root/Path(x).relative_to("./CheXpert-v1.0-small" if chexphoto is False else "./CheXphoto-v1.0"))
    
    #patients
    df_train["patient_id"]=df_train.path.apply(lambda x: x.parts[-3])
    df_test["patient_id"]=df_test.path.apply(lambda x: x.parts[-3])

    #ids
    df_train["image_id"]=df_train.path.apply(lambda x: (("_".join(x.parts[-3:]))[:-4]).replace("patient","").replace("study","").replace("view",""))
    df_test["image_id"]=df_test.path.apply(lambda x: (("_".join(x.parts[-3:]))[:-4]).replace("patient","").replace("study","").replace("view",""))

    if(chexphoto):
        #category-subcategory
        #synthetic-digital/photographic
        #natural-oneplus (test)
        #natural-iphone/nokia (train)
        df_train["chexphoto_category"]=df_train.Path.apply(lambda x: Path(x).parts[2])
        df_train["chexphoto_subcategory"]=df_train.Path.apply(lambda x: Path(x).parts[3])
        df_test["chexphoto_category"]=df_test.Path.apply(lambda x: Path(x).parts[2])
        df_test["chexphoto_subcategory"]=df_test.Path.apply(lambda x: Path(x).parts[3])

    if(full_dataset_as_test):
       print("Warning: using the full dataset for testing")
       df_test = pd.concat([df_train,df_test])
            
    #potentially select only frontal/lateral images
    if(select_frontal_lateral_train=="Frontal" or select_frontal_lateral_train=="Lateral"):
        df_train = df_train[df_train["Frontal/Lateral"]==select_frontal_lateral_train]
    if(select_frontal_lateral_test=="Frontal" or select_frontal_lateral_test=="Lateral"):
        df_test = df_test[df_test["Frontal/Lateral"]==select_frontal_lateral_test]
    
    print(root,"CheXpert train:" if chexphoto is False else "CheXphoto train:",len(df_train),"samples.")
    print(root,"CheXpert test:" if chexphoto is False else "CheXphoto test:",len(df_test),"samples.")
    
    return df_train, df_test, label_itos

def prepare_mimic_cxr_df(root=Path("."), 
                         label_itos = None,
                         use_chexpert_labels=True):
    '''prepares dataframe for the mimic cxr dataset'''
    if(label_itos is None):
        if(use_chexpert_labels):
            label_itos = label_itos_chexpert14
        else:
            label_itos = label_itos_cxr14
    root = Path(root)/"mimic-cxr-jpg-2.0.0-small"
    print("Preparing mimic_cxr_df...")
    
    def merge_mimic_cxr_df(root_path=Path("."), use_chexpert_labels=True):
        if(use_chexpert_labels):
            df_label = pd.read_csv(root_path/"mimic-cxr-2.0.0-chexpert.csv")
            df_label["has_label"] = True
            df_label = df_label.set_index(["subject_id","study_id"])
        else:#negbio
            df_label = pd.read_csv(root_path/"mimic-cxr-2.0.0-negbio.csv")
            df_label["has_label"] = True
            df_label = df_label.set_index(["subject_id","study_id"])
        df_meta = pd.read_csv(root_path/"mimic-cxr-2.0.0-metadata.csv")
        df_splits = pd.read_csv(root_path/"mimic-cxr-2.0.0-split.csv")
        df_splits=df_splits.set_index("dicom_id")
        df_mimic_cxr = df_meta.join(df_splits[["split"]],on="dicom_id",how="left").join(df_label,how="left",on=["subject_id","study_id"])
        df_mimic_cxr.has_label=df_mimic_cxr.has_label.apply(lambda x: x is True)
        def add_path(subject_id,study_id,dicom_id):
            return root_path/"files_small"/("p"+str(subject_id)[:2])/("p"+str(subject_id))/("s"+str(study_id))/(dicom_id+".jpg")
        df_mimic_cxr["path"]=df_mimic_cxr.apply(lambda row: add_path(row["subject_id"],row["study_id"],row["dicom_id"]),axis=1)
        return df_mimic_cxr

    df_mimic_cxr = merge_mimic_cxr_df(root_path=root, use_chexpert_labels=use_chexpert_labels)
    
    #print("Mapping labels...")
    label_map_uones={0.0:0, 1.0:1, -1.0:1}
    label_map_uzeros={0.0:0, 1.0:1, -1.0:0}
    label_map_raw={0.0:0, 1.0:1, -1.0:-1}

    def apply_map(x,mp):
        if(np.isnan(x)):#set unmentioned to 0
            return 0.
        else:
            return mp[x]
        
    #multi-hot encoded labels
    #print("uones...")
    df_mimic_cxr["label_uones"]=df_mimic_cxr.apply(lambda row: np.array([apply_map(row[c],label_map_uones) for c in label_itos]).astype(np.float32),axis=1)
    #print("uzeros...")
    df_mimic_cxr["label_uzeros"]=df_mimic_cxr.apply(lambda row: np.array([apply_map(row[c],label_map_uzeros) for c in label_itos]).astype(np.float32),axis=1)
    #print("raw...")
    df_mimic_cxr["label_raw"]=df_mimic_cxr.apply(lambda row: np.array([apply_map(row[c],label_map_raw) for c in label_itos]).astype(np.float32),axis=1)
    df_mimic_cxr["label"]=df_mimic_cxr["label_uones"] #default label
    df_mimic_cxr["patient_id"]=df_mimic_cxr.subject_id #for consistency with chexpert dataset
    df_mimic_cxr["image_id"]=df_mimic_cxr.dicom_id #id for consistency
    
    #print("done.")
    df_train = df_mimic_cxr[df_mimic_cxr.split=="train"]
    df_valid = df_mimic_cxr[df_mimic_cxr.split=="validate"]
    df_test = df_mimic_cxr[df_mimic_cxr.split=="test"]
    
    print(root,"MIMIC-CXR train:",len(df_train),"samples.")
    print(root,"MIMIC-CXR valid:",len(df_valid),"samples.")
    print(root,"MIMIC-CXR test:",len(df_test),"samples.")
    
    return df_train, df_valid, df_test, label_itos

def prepare_cxr14_df(root=Path("."), 
                         label_itos = label_itos_cxr14,
                         use_small_version=True):
    '''prepares dataframe for the chestxray8/14 dataset dataset'''
    root = Path(root)/"cxr14-small"
    print("Preparing cxr14_df...")
    def merge_cxr14_df(root_path=Path("."), use_small_version=True):
        df_train_val = pd.read_table(root_path/"train_val_list.txt",header=None)
        df_train_val["train_set"]=True
        df_train_val=df_train_val.set_index(0)
        df_test = pd.read_table(root_path/"test_list.txt",header=None)
        df_test["test_set"]=True
        df_test=df_test.set_index(0)
        df= pd.read_csv(root_path/"Data_Entry_2017_v2020.csv")

        for l in label_itos:
            df[l] = df["Finding Labels"].apply(lambda x: l in x)
        df=df.join(df_train_val,on="Image Index",how="left").join(df_test,on="Image Index", how="left")
        df["train_set"]=df["train_set"].apply(lambda x: x is True)
        df["test_set"]=df["test_set"].apply(lambda x: x is True)
        df["split"]=df.apply(lambda row: "train" if row["train_set"] is True else ("test" if row["test_set"] is True else "none"),axis=1)
        df["image_id"]=df["Image Index"].apply(lambda x: x.split(".")[0])
        df["patient_id"]=df["Patient ID"]#for consistency reasons
        
        res = []
        for f in list((Path(root_path)/"images_small").glob('**/*.jpg' if use_small_version else '**/*.png')):
            res.append({"image_id":f.stem, "path":f})
        df_files = pd.DataFrame(res).set_index("image_id")
        df = df.join(df_files,how="left",on="image_id")
        return df

    df_cxr14 = merge_cxr14_df(root_path=root, use_small_version= use_small_version)
    
    #print("Mapping labels...")
    label_map_uones={0.0:0, 1.0:1, -1.0:1}
    label_map_uzeros={0.0:0, 1.0:1, -1.0:0}
    label_map_raw={0.0:0, 1.0:1, -1.0:-1}

    def apply_map(x,mp):
        if(np.isnan(x)):#set unmentioned to 0
            return 0.
        else:
            return mp[x]
        
    #multi-hot encoded labels
    #print("uones...")
    df_cxr14["label_uones"]=df_cxr14.apply(lambda row: np.array([apply_map(row[c],label_map_uones) for c in label_itos]).astype(np.float32),axis=1)
    #print("uzeros...")
    df_cxr14["label_uzeros"]=df_cxr14.apply(lambda row: np.array([apply_map(row[c],label_map_uzeros) for c in label_itos]).astype(np.float32),axis=1)
    #print("raw...")
    df_cxr14["label_raw"]=df_cxr14.apply(lambda row: np.array([apply_map(row[c],label_map_raw) for c in label_itos]).astype(np.float32),axis=1)
    df_cxr14["label"]=df_cxr14["label_uones"] #default label
    #print("done.")
    df_train = df_cxr14[df_cxr14.split=="train"]
    df_test = df_cxr14[df_cxr14.split=="test"]
    
    print(root,"CXR14 train:",len(df_train),"samples.")
    print(root,"CXR14 test:",len(df_test),"samples.")
    
    return df_train, df_test, label_itos

##############################################################################
#DIABETIC RETINOPATHY
##############################################################################

def prepare_diabetic_retinopathy_df(root, args, label_itos = [*map(str,range(5))]):
    root = Path(root)

    df_train = pd.read_csv(root/"trainLabels.csv")
    df_train["patient"]=df_train["image"].apply(lambda x: x.split("_")[0])
    df_train["left_eye"]=df_train["image"].apply(lambda x: x.split("_")[1]=="left")
    df_train["label"]=df_train["level"]
    df_train["path"]=df_train["image"].apply(lambda x:root/"train"/(x+".jpeg"))

    df_valid = pd.read_csv(root/"retinopathy_solution.csv")
    df_valid["patient"]=df_valid["image"].apply(lambda x: x.split("_")[0])
    df_valid["left_eye"]=df_valid["image"].apply(lambda x: x.split("_")[1]=="left")
    df_valid["label"]=df_valid["level"]
    df_valid["path"]=df_valid["image"].apply(lambda x:root/"test"/(x+".jpeg"))

    ### Randomly select 33423 samples from test set
    import random
    random.seed(1)
    valid_idxs = random.sample(range(len(df_train)), 5126)
    train_idxs = list(set(range(len(df_train))) - set(valid_idxs))
    
    df_test = df_valid
    df_valid = df_train.loc[valid_idxs]
    df_train = df_train.loc[train_idxs]
    
    print(root,"train:",len(df_train),"samples.")    
    print(root,"valid:",len(df_valid),"samples.")
    print(root,"test:",len(df_test),"samples.")
    
    return df_train, df_valid, df_test, label_itos

def prepare_binary_classification(df, eval_dr='binary_rdr'):
    if eval_dr == 'binary_rdr':
        df["label"].replace(
            to_replace=[0, 1, 2],
            value=0,
            inplace=True
        )   
        df["label"].replace(
            to_replace=[3, 4],
            value=1,
            inplace=True
        )
        
    elif eval_dr == 'binary_norm':
        df["label"].replace(
            to_replace=[2, 3, 4],
            value=1,
            inplace=True
        )   
    elif eval_dr == 'binary_dme':
        df["label"].replace(
            to_replace=[0, 1],
            value=0,
            inplace=True
        )   

        df["label"].replace(
            to_replace=[2, 3, 4],
            value=1,
            inplace=True
        )

    return df

def prepare_messidor_df(root, label_itos = [*map(str,range(5))]):
    root = Path(root)

    df_test = pd.read_csv(root/"messidor_data.csv")
    # import pdb;
    # pdb.set_trace()
    df_test["patient"]="IMAGES/"+df_test["image_id"]
    df_test["label"]=df_test["adjudicated_dr_grade"]
    #df_test["binary_label"]=df_test["adjudicated_dme"] # Referrable vs Non-Referrable
    df_test["path"]=df_test["image_id"].apply(lambda x:root/"IMAGES"/x)

    shape = df_test.shape[0]
    df_test = df_test[df_test['adjudicated_dr_grade'].notna()]
    print("Number of test samples removed: {}".format(df_test.shape[0] - shape))

    df_train = df_test.copy() #dummy
    
    print(root,"test:",len(df_test),"samples.")

    return df_train, df_test, label_itos

def prepare_messidor_1_df(root, label_itos = [*map(str,range(5))]):
    root = Path(root)

    df_test = pd.read_excel(root/"test/test.xls")
    # import pdb;
    # pdb.set_trace()
    df_test["patient"]="test/"+df_test["Image name"]
    df_test["label"]=df_test["Retinopathy grade"]
    #df_test["binary_label"]=df_test["adjudicated_dme"] # Referrable vs Non-Referrable
    df_test["path"]=df_test["Image name"].apply(lambda x:root/"test"/x)

    df_train = pd.read_excel(root/"train/train.xls")
    # import pdb;
    # pdb.set_trace()
    df_train["patient"]="train/"+df_train["Image name"]
    df_train["label"]=df_train["Retinopathy grade"]
    #df_test["binary_label"]=df_test["adjudicated_dme"] # Referrable vs Non-Referrable
    df_train["path"]=df_train["Image name"].apply(lambda x:root/"train"/x)

    df_test = pd.concat([df_train,df_test],ignore_index=True)
    
    print(root,"train:",len(df_train),"samples.")
    print(root,"test:",len(df_test),"samples.")
    
    return df_train, df_test, label_itos

##############################################################################
#STL-10
##############################################################################
def target_transform_stl10(x):
    return [0, 0] if x==-1 else [x,1]
##############################################################################
#GETTING DATATSET STATS
##############################################################################
def stats_from_ds(ds,div=255):
    means=[]
    stds=[]
    for d in tqdm(ds_train):
        stat = Stat(d[0])
        means.append(stat.mean)
        stds.append(stat.stddev)
    means = np.mean(means,axis=0)
    stds = np.mean(stds,axis=0)
    return [means/div,stds/div]

#chexpert_stats = stats_from_ds(ds_train)
