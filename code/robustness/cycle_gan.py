from tqdm import tqdm
from argparse import ArgumentParser
import numpy as np, pandas as pd,  matplotlib as mpl, matplotlib.pyplot as plt,  os
import itertools, functools
from skimage import io as io,  transform as tfm
from torchvision.datasets.folder import default_loader
import json
import torch, torch.nn as nn, torch.nn.functional as F,  torch.optim as optim
import torchvision,  torchvision.transforms as T,  torchvision.utils as utils
from torch.nn import Conv2d as Conv, ConvTranspose2d as Deconv,  ReLU as Relu
from torch.nn import InstanceNorm2d as InstanceNorm, BatchNorm2d as BatchNorm
from torch.nn.init import kaiming_uniform_
from torch.nn.utils import weight_norm
from torch.utils.tensorboard import SummaryWriter,  FileWriter,  RecordWriter
from torch.utils.data import Dataset, DataLoader, ConcatDataset, TensorDataset
from nnunet.network_architecture.generic_UNet import Generic_UNet

import glob 
import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import main_patho_lightning
from utils.dataloader import *


class TrafoWrapper:

    def __init__(self, f_trafo):
        self.f = f_trafo

    def __call__(self, sample):
        return {k: self.f(sample[k]) for k in 'AB'}


class Resize(object):
    
    def __init__(self, image_size: (int, tuple) = 256):
        
        """
        Parameters:
            image_size: Final size of the image
        """
        
        if   isinstance(image_size, int):   self.image_size = (image_size, image_size)
        elif isinstance(image_size, tuple): self.image_size = image_size
        else: raise ValueError("Unknown DataType of the parameter image_size found!!")
      
    
    def __call__(self, sample):
        
        """
        Parameters: 
            sample: Dictionary containing image and label
        """
        
        A, B = sample['A'], sample['B']
        
        A = tfm.resize(A, output_shape = self.image_size)
        B = tfm.resize(B, output_shape = self.image_size)
        
        A = np.clip(A, a_min = 0., a_max = 1.)
        B = np.clip(B, a_min = 0., a_max = 1.)
        
        return {'A': A, 'B': B}


class RandomCrop(object):
    
    def __init__(self, image_size: (int, tuple) = 256): 
        
        """
        Parameters: 
            image_size: Final size of the image (should be smaller than current size o/w 
                        returns the original image)
        """
        
        if   isinstance(image_size, int):   self.image_size = (image_size, image_size)
        elif isinstance(image_size, tuple): self.image_size = image_size
        else: raise ValueError("Unknown DataType of the parameter image_size found!!")
       
    
    def __call__(self, sample):
        
        """
        Parameters: 
            sample: Dictionary containing image and label
        """
        
        A, B = sample['A'], sample['B']
        curr_height, curr_width = A.shape[0], A.shape[1]
        
        ht_diff = max(0, curr_height - self.image_size[0])
        wd_diff = max(0, curr_width  - self.image_size[1])
        top = np.random.randint(low = 0, high = ht_diff)
        lft = np.random.randint(low = 0, high = wd_diff)
        
        A = A[top: top + self.image_size[0], lft: lft + self.image_size[1]]
        B = B[top: top + self.image_size[0], lft: lft + self.image_size[1]]
        
        return {'A': A, 'B': B}
    

class Random_Flip(object):
    
    def __call__(self, sample):
        
        """
        Parameters: 
            sample: Dictionary containing image and label
        """
        
        A, B = sample['A'], sample['B']
        if np.random.uniform(low = 0., high = 1.0) > .5:
            A = np.fliplr(A); B = np.fliplr(B)
        
        return {'A': A, 'B': B}


class To_Tensor(object):
    
    def __call__(self, sample):
        
        """
        Parameters: 
            sample: Dictionary containing image and label
        """
        
        A = np.transpose(sample['A'].astype(np.float, copy = True), (2, 0, 1))
        B = np.transpose(sample['B'].astype(np.float, copy = True), (2, 0, 1))
        
        A = torch.tensor(A, dtype = torch.float)
        B = torch.tensor(B, dtype = torch.float)
        
        return {'A': A, 'B': B}
    

class Normalize(object):
    
    def __init__(self, mean = [0.5] * 3, stdv = [0.5] * 3):
        
        """
        Parameters: 
            mean: Normalizing mean
            stdv: Normalizing stdv
        """
        
        mean = torch.tensor(mean, dtype = torch.float)
        stdv = torch.tensor(stdv, dtype = torch.float)
        self.transforms = T.Normalize(mean = mean, std = stdv)
     
    
    def __call__(self, sample):
        
        """
        Parameters:
            sample: Dictionary containing image and label
        """
        
        A, B = sample['A'], sample['B']
        A = self.transforms(A)
        B = self.transforms(B)
        
        return {'A': A, 'B': B}


class CustomDataset(Dataset):
    
    def __init__(self, pathA: str = None, pathB: str = None, transforms = None):
        
        """Creates dataset object for the two datasets given by the paths. 

        Args:
            pathA (string): Path to first of the transformer datasets
            pathB (string): Path to second of the transformer datasets
            transforms (list of torch transforms): a list of Transformations (Data augmentation)
        Returns:
            [array]: Array of length 3 with RGB mean values as floats
        """
        
        super().__init__(); self.transforms = T.Compose(transforms)
        
        #file_names_A = sorted(os.listdir(pathA), key = lambda x: int(x[: -4]))
        #file_names_A = sorted(os.listdir(pathA), key = str.lower)

        #self.file_names_A = [pathA + file_name for file_name in file_names_A]
        self.file_names_A = self.__search_imgs(pathA)
        
        #file_names_B = sorted(os.listdir(pathB), key = lambda x: int(x[: -4]))
        #file_names_B = sorted(os.listdir(pathB), key = str.lower)
        #self.file_names_B = [pathB + file_name for file_name in file_names_B]
        self.file_names_B = self.__search_imgs(pathB)

    def __search_imgs(self, dset_root, suffixes=['.jpg', '.png', '.jpeg', '.tiff']):
        suffixes += [sfx.upper() for sfx in suffixes]
        files = []
        for r, _, fs in os.walk(dset_root):
            for f in fs:
                if any([f.endswith(sfx) for sfx in suffixes]):
                    files.append(os.path.join(r, f))
        return files
        
        
    def __len__(self):
        # why min ????
        #return min(len(self.file_names_A), len(self.file_names_B))
        return max(len(self.file_names_A), len(self.file_names_B))
    
    
    def __getitem__(self, idx):
        
        # skimage imread is not as fast...
        # A = io.imread(self.file_names_A[idx % len(self.file_names_A)])
        # B = io.imread(self.file_names_B[idx % len(self.file_names_B)])
        A = default_loader(self.file_names_A[idx % len(self.file_names_A)])
        B = default_loader(self.file_names_B[idx % len(self.file_names_B)])
        sample = self.transforms({'A': A, 'B': B})
        
        return sample


class Helper(object):
    
    @staticmethod
    def show_image(image):
        
        image = np.transpose((image + 1) / 2, (1, 2, 0))
        plt.imshow(image)
        
        
    @staticmethod
    def tensor_to_numpy(tensor):
        
        tensor = (tensor.cpu().clone() + 1) / 2
        if   len(tensor.shape) == 3: tensor = np.transpose(tensor, (1, 2, 0))
        elif len(tensor.shape) == 4: tensor = np.transpose(tensor, (0, 2, 3, 1))
        
        return tensor

    
    @staticmethod
    def get_random_sample(dataset):
        return dataset[np.random.randint(0, len(dataset))]
    
    
    @staticmethod
    def get_data(pathA: str, pathB: str, tfms, batch_sz: int, is_train: bool):

        print('... getting the data')
        
        dataset = CustomDataset(pathA = pathA, pathB = pathB, transforms = tfms)
        print('dataset loaded')
        dataloader = DataLoader(dataset, batch_size = batch_sz, shuffle = is_train, num_workers = 4)
        print('dataloader ready')
        return dataset, dataloader


class ResBlock(nn.Module):
    
    def __init__(self, in_channels: int, apply_dp: bool = True):
        
        """
                            Defines a ResBlock
        X ------------------------identity------------------------
        |-- Convolution -- Norm -- ReLU -- Convolution -- Norm --|
        """
        
        """
        Parameters:
            in_channels:  Number of input channels
            apply_dp:     If apply_dp is set to True, then activations are 0'ed out with prob 0.5
        """
        
        super().__init__()
        
        conv = nn.Conv2d(in_channels = in_channels, out_channels = in_channels, kernel_size = 3, stride = 1)
        layers =  [nn.ReflectionPad2d(1), conv, nn.InstanceNorm2d(in_channels), nn.ReLU(True)]
        
        if apply_dp: layers += [nn.Dropout(0.5)]
        
        conv = nn.Conv2d(in_channels = in_channels, out_channels = in_channels, kernel_size = 3, stride = 1)
        layers += [nn.ReflectionPad2d(1), conv, nn.InstanceNorm2d(in_channels)]
        
        self.net = nn.Sequential(*layers)
    
    
    def forward(self, x): return x + self.net(x)


class Interpolate(nn.Module):

    def __init__(self, scale_factor, mode):

        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode
        
    def forward(self, x):
        x = self.interp(x, scale_factor=self.scale_factor, mode=self.mode)
        return x


class ConvLayer(nn.Sequential):
    "Create a sequence of convolutional (`ni` to `nf`), ReLU (if `use_activ`) and `norm_type` layers."
    def __init__(self, ni, nf, ks=3, stride=1, padding=0, ndim=2, bn_1st=True,
                 act_cls=nn.ReLU, transpose=False, init='auto', xtra=None, bias_std=0.01, **kwargs):
        
        conv = nn.Conv2d(in_channels=ni, out_channels=nf, kernel_size=ks, stride=stride, padding=padding, **kwargs)
        act = None if act_cls is None else act_cls()
        conv = weight_norm(conv)
        layers = [conv]
        act_bn = []
        if act is not None: act_bn.append(act)
        layers += act_bn

        super().__init__(*layers)


def icnr_init(x, scale=2, init=nn.init.kaiming_normal_):
    "ICNR init of `x`, with `scale` and `init` function"
    ni,nf,h,w = x.shape
    ni2 = int(ni/(scale**2))
    k = init(x.new_zeros([ni2,nf,h,w])).transpose(0, 1)
    k = k.contiguous().view(ni2, nf, -1)
    k = k.repeat(1, 1, scale**2)
    return k.contiguous().view([nf,ni,h,w]).transpose(0, 1)


class PixelShuffle_ICNR(nn.Sequential):
    "Upsample by `scale` from `ni` filters to `nf` (default `ni`), using `nn.PixelShuffle`."
    def __init__(self, ni, nf=None, scale=2, act_cls=nn.ReLU):
        super().__init__()
        nf = ni if nf is None else nf
        layers = [ConvLayer(ni, nf*(scale**2), ks=1, act_cls=act_cls, bias_std=0),
                  nn.PixelShuffle(scale)]
        layers[0][0].weight_v.data.copy_(icnr_init(layers[0][0].weight_v.data))
        layers[0][0].weight_g.data.copy_(((layers[0][0].weight_v.data**2).sum(dim=[1,2,3])**0.5)[:,None,None,None])
        super().__init__(*layers)


class Generator(nn.Module):
    
    def __init__(self, in_channels: int = 3, out_channels: int = 64, apply_dp: bool = True):
        
        """
                                Generator Architecture (Image Size: 128)
        c7s1-64, d128, d256, R256, R256, R256, R256, R256, R256, u128, u64, c7s1-3, 
        
        where c7s1-k denote a 7 × 7 Conv-InstanceNorm-ReLU layer with k filters and stride 1, dk denotes a 3 × 3
        Conv-InstanceNorm-ReLU layer with k filters and stride 2, Rk denotes a residual block that contains two 
        3 × 3 Conv layers with the same number of filters on both layer. uk denotes a 3 × 3 DeConv-InstanceNorm-
        ReLU layer with k filters and stride 1.
        """
        
        """
        Parameters: 
            in_channels:  Number of input channels 
            out_channels: Number of output channels
            apply_dp:     If apply_dp is set to True, then activations are 0'ed out with prob 0.5
        """
        
        super().__init__()
        
        f = 1
        nb_downsampling = 2
        #nb_resblks = 6 if img_sz == 128 else 9 
        nb_resblks = 3
        
        conv = nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = 7, stride = 1)
        self.layers = [nn.ReflectionPad2d(3), conv, nn.InstanceNorm2d(out_channels), nn.ReLU(True)]
        
        for i in range(nb_downsampling):
            conv = nn.Conv2d(out_channels * f, out_channels * 2 * f, kernel_size = 3, stride = 2, padding = 1)
            #self.layers += [conv, nn.InstanceNorm2d(out_channels * 2 * f), nn.ReLU(True)]
            self.layers += [conv, nn.InstanceNorm2d(out_channels * 2 * f), nn.LeakyReLU(negative_slope=0.2, inplace=True)]
            f *= 2
        
        for i in range(nb_resblks):
            res_blk = ResBlock(in_channels = out_channels * f, apply_dp = apply_dp)
            self.layers += [res_blk]
        
        for i in range(nb_downsampling):

            ####### original transpose2d #######
            #conv = nn.ConvTranspose2d(out_channels * f, out_channels * (f//2), 3, 2, padding = 1, output_padding = 1)

            ####### Pixelshuffle #######
            #conv = nn.PixelShuffle(2)
            #conv = PixelShuffle_ICNR(out_channels*f, out_channels*(f//2), scale=2)

            ####### Interpolation #######
            intp = Interpolate(scale_factor=2, mode='nearest')
            conv = nn.Conv2d(in_channels = out_channels*f, out_channels = out_channels*(f//2), 
                               kernel_size = 3, stride=1, padding=1, padding_mode='replicate')
            
            #self.layers += [conv, nn.InstanceNorm2d(out_channels * (f//2)), nn.ReLU(True)]
            
            # for interpolation:
            self.layers += [intp, conv, nn.InstanceNorm2d(out_channels * (f//2)), nn.ReLU(True)]
            
            f = f // 2
        
        # for transposed2 in_channels = out_channels
        conv = nn.Conv2d(in_channels = out_channels, out_channels = in_channels, kernel_size = 7, stride = 1)
        self.layers += [nn.ReflectionPad2d(3), conv, nn.Tanh()]
        
        self.net = nn.Sequential(*self.layers)   
    
    def forward(self, x): 

        # for one skip connection in the generator
        # from input to output
        ipt = x

        for layer in self.net:
            x = layer(x)
            """ print(layer)
            print(x.size()) """
        
        #return self.net(x)
        return (self.net(x) + ipt)/2


class Mean2d(nn.Module):

    def forward(self, x):
        return x.mean(dim=-1).mean(dim=-1)


class Discriminator(nn.Module):
    
    def __init__(self, in_channels: int = 3, out_channels: int = 64, nb_layers: int = 3, global_discriminator=True):
        
        """
                                    Discriminator Architecture!
        C64 - C128 - C256 - C512, where Ck denote a Convolution-InstanceNorm-LeakyReLU layer with k filters
        """
        
        """
        Parameters: 
            in_channels:    Number of input channels
            out_channels:   Number of output channels
            nb_layers:      Number of layers in the 70*70 Patch Discriminator
        """
        
        super().__init__()
               

        if global_discriminator:
            conv = lambda c_in, c_out, ks, act: nn.Sequential(
                nn.Conv2d(c_in,  c_out, kernel_size=ks, padding=int(ks/2)),
                nn.BatchNorm2d(c_out),
                act
            )
            f_relu = nn.ReLU
            f_sigmoid = nn.Sigmoid
            features = 3
            self.layers = [conv(in_channels, features, 3, f_relu()), conv(features, features, 3, f_relu()), conv(features, features, 1, f_sigmoid()), Mean2d()]
        else:
            in_f  = 1
            out_f = 2
            
            conv = nn.Conv2d(in_channels, out_channels, kernel_size = 4, stride = 2, padding = 1)
            self.layers = [conv, nn.LeakyReLU(0.2, True)]
            
            conv = lambda : nn.Conv2d(out_channels * in_f, out_channels * out_f, kernel_size = 4, stride = 2, padding = 1)
            for idx in range(1, nb_layers):
                self.layers += [conv(), nn.InstanceNorm2d(out_channels * out_f), nn.LeakyReLU(0.2, True)]
                in_f   = out_f
                out_f *= 2
            out_f = min(2 ** nb_layers, 8)
            conv = lambda : nn.Conv2d(out_channels * in_f,  out_channels * out_f, kernel_size = 4, stride = 1, padding = 1)
            self.layers += [conv(), nn.InstanceNorm2d(out_channels * out_f), nn.LeakyReLU(0.2, True)]      
            conv = lambda: nn.Conv2d(out_channels * out_f, out_channels = 1, kernel_size = 4, stride = 1, padding = 1)
            self.layers += [conv()]
        
        self.net = nn.Sequential(*self.layers)
        
    def forward(self, x): 
        return self.net(x)
    

class Initializer:
    
    def __init__(self, init_type: str = 'normal', init_gain: float = 0.02): 
        
        """
        Initializes the weight of the network!
        
        Parameters: 
            init_type: Initializer type - 'kaiming' or 'xavier' or 'normal'
            init_gain: Standard deviation of the normal distribution
        """
        
        self.init_type = init_type; self.init_gain = init_gain
        
        
    def init_module(self, m):
        
        cls_name = m.__class__.__name__;
        if hasattr(m, 'weight') and (cls_name.find('Conv') != -1 or cls_name.find('Linear') != -1):
            
            if   self.init_type == 'kaiming': nn.init.kaiming_normal_(m.weight.data, a = 0, mode = 'fan_in')
            elif self.init_type == 'xavier' : nn.init.xavier_normal_ (m.weight.data,  gain = self.init_gain)
            elif self.init_type == 'normal' : nn.init.normal_(m.weight.data, mean = 0, std = self.init_gain)
            else: raise ValueError('Initialization not found!!')
            
            if m.bias is not None: nn.init.constant_(m.bias.data, val = 0); 
            
        if hasattr(m, 'weight') and cls_name.find('BatchNorm2d') != -1:
            nn.init.normal_(m.weight.data, mean = 1.0, std = self.init_gain)
            nn.init.constant_(m.bias.data, val = 0)
            
            
    def __call__(self, net):
        
        """
        Parameters: 
            net: Network
        """
        
        net = net.to(devices[0]) 
        net = nn.DataParallel(net, device_ids = range(len(devices)))
        net.apply(self.init_module)
        
        return net


class Tensorboard:
    
    def __init__(self, path: str): self.writer = SummaryWriter(path)
    
    
    def write_graph(self, model): 
        
        A = helper.get_random_sample(trn_dataset)['A'].unsqueeze(0)
        self.writer.add_graph(model.module, A.to(devices[0]))
    
    
    @torch.no_grad()
    def write_image(self, nb_examples: int, g_A2B, g_B2A, epoch: int, curr_iter: int):
        
        grid_A = []
        grid_B = []
        n_iter = (epoch - 1) * nb_trn_iters + curr_iter
        
        for _ in range(nb_examples):
            
            sample = helper.get_random_sample(val_dataset)
            real_A = sample['A'].unsqueeze(0).to(devices[0])
            real_B = sample['B'].unsqueeze(0).to(devices[0])

            fake_A = g_B2A(real_B); cyc_B = g_A2B(fake_A)
            fake_B = g_A2B(real_A); cyc_A = g_B2A(fake_B)

            tensor = torch.cat([real_A, fake_B, cyc_A, real_B, fake_A, cyc_B])
            tensor = (tensor.cpu().clone() + 1) / 2
            
            grid_A.append(tensor[:3])
            grid_B.append(tensor[3:]) 
        
        grid_A = torchvision.utils.make_grid(torch.cat(grid_A, 0), nrow = 6)
        grid_B = torchvision.utils.make_grid(torch.cat(grid_B, 0), nrow = 6)

        self.writer.add_image('Grid_A', grid_A, n_iter)
        self.writer.add_image('Grid_B', grid_B, n_iter)
        
    
    @torch.no_grad()
    def write_loss(self, d_loss: float, g_loss: float, epoch: int, curr_iter: int):
        
        n_iter = (epoch - 1) * nb_trn_iters + curr_iter
        
        self.writer.add_scalar('d_loss', round(d_loss.item(), 4), n_iter)
        self.writer.add_scalar('g_loss', round(g_loss.item(), 4), n_iter)



class Loss:
    
    """
    This class implements different losses required to train the generators and discriminators of CycleGAN
    """
    
    def __init__(self, loss_type: str = 'MSE', lambda_: int = 10):
        
        """
        Parameters:
            loss_type: Loss Function to train CycleGAN
            lambda_:   Weightage of Cycle-consistency loss
        """
        
        self.loss = nn.MSELoss() if loss_type == 'MSE' else nn.BCEWithLogitsLoss()
        self.lambda_ = lambda_
        
    
    def get_dis_gan_loss(self, dis_pred_real_data, dis_pred_fake_data):
        
        """
        Parameters:
            dis_pred_real_data: Discriminator's prediction on real data
            dis_pred_fake_data: Discriminator's prediction on fake data
        """
        
        dis_tar_real_data = torch.ones_like (dis_pred_real_data, requires_grad = False)
        dis_tar_fake_data = torch.zeros_like(dis_pred_fake_data, requires_grad = False)
        
        loss_real_data = self.loss(dis_pred_real_data, dis_tar_real_data)
        loss_fake_data = self.loss(dis_pred_fake_data, dis_tar_fake_data)
        
        dis_tot_loss = (loss_real_data + loss_fake_data) * 0.5
        
        return dis_tot_loss
    
    
    def get_gen_gan_loss(self, dis_pred_fake_data):
        
        """
        Parameters:
            dis_pred_fake_data: Discriminator's prediction on fake data
        """
        
        gen_tar_fake_data = torch.ones_like(dis_pred_fake_data, requires_grad = False)
        gen_tot_loss = self.loss(dis_pred_fake_data, gen_tar_fake_data)
        
        return gen_tot_loss
    
    
    def get_gen_cyc_loss(self, real_data, cyc_data):
        
        """
        Parameters:
            real_data: Real images sampled from the dataloaders
            cyc_data:  Image reconstructed after passing the real image through both the generators
                       X_recons = F * G (X_real), where F and G are the two generators
        """
        
        gen_cyc_loss = torch.nn.L1Loss()(real_data, cyc_data)
        gen_tot_loss = gen_cyc_loss * self.lambda_
        
        return gen_tot_loss
    
    
    def get_gen_idt_loss(self, real_data, idt_data):
        
        """
        Implements the identity loss: 
            nn.L1Loss(LG_B2A(real_A), real_A) 
            nn.L1Loss(LG_A2B(real_B), real_B) 
        """
        
        gen_idt_loss = torch.nn.L1Loss()(real_data, idt_data)
        gen_tot_loss = gen_idt_loss * self.lambda_ * 0.5
        
        return gen_tot_loss



class ImagePool:
    
    """
    This class implements an image buffer that stores previously generated images! This buffer enables to update
    discriminators using a history of generated image rather than the latest ones produced by generator.
    """
    
    def __init__(self, pool_sz: int = 50):
        
        """
        Parameters:
            pool_sz: Size of the image buffer
        """
        
        self.pool_sz = pool_sz
        self.image_pool = []
        self.nb_images = 0
        
    def empty(self):
        self.image_pool = []
        self.nb_images = 0
    
    def push_and_pop(self, images):
        
        """
        Parameters:
            images: latest images generated by the generator
        
        Returns a batch of images from pool!
        """
        images_to_return = []
        for image in images:
            image = torch.unsqueeze(image, 0)#.detach().cpu()
            if  len(self.image_pool) < self.pool_sz:
                self.image_pool.append (image) 
                images_to_return.append(image)
            else:
                #if np.random.uniform(0, 1) > 0.5:
                rand_int = np.random.randint(0, self.pool_sz)
                temp_img = self.image_pool[rand_int].clone()
                self.image_pool[rand_int] = image
                images_to_return.append(temp_img)   
                #else:
                #    images_to_return.append(image)
        
        images_to_return = torch.cat(images_to_return, 0)
        return images_to_return



class SaveModel:
    
    def __init__(self, path: str, keep_only: int = 50): 
        
        self.path = path
        self.keep_only = keep_only
    
    
    def save_model(self, epoch: int, d_A, d_B, g_A2B, g_B2A, d_opt, g_opt):
        
        filename = self.path + "Model_" + str(epoch) + ".pth"
        
        torch.save({'epochs': epoch, 'd_opt': d_opt.state_dict(), 'g_opt': g_opt.state_dict(), 
                    'd_A': d_A.module.state_dict(), 'd_B': d_B.module.state_dict(),
                    'g_A2B': g_A2B.module.state_dict(), 'g_B2A': g_B2A.module.state_dict()}, filename)
        
        
        filenames = [f for f in os.listdir(self.path) if not f.startswith('.')]
        if len(filenames) > self.keep_only:
            os.remove(self.path + sorted(filenames, key = lambda x: int(x[6 : -4]))[0])



class CycleGAN:
    
    def __init__(self, root_dir: str, g_A2B, g_B2A, d_A, d_B):
        
        self.save_dir = root_dir + 'Models/'
        summary_path = root_dir + 'Tensorboard/'
        
        if not os.path.exists(self.save_dir): os.makedirs(self.save_dir)
        if not os.path.exists(summary_path ): os.makedirs(summary_path )
        self.saver = SaveModel(self.save_dir); self.tb = Tensorboard(summary_path)
        
        self.d_A = d_A
        self.d_B = d_B
        self.g_A2B = g_A2B
        self.g_B2A = g_B2A
        self.fake_pool_A = ImagePool(pool_sz = 50)
        self.fake_pool_B = ImagePool(pool_sz = 50)
        self.loss = Loss(loss_type = 'MSE', lambda_ = 10)
        self.trn_dataloader = None
        
        
    def load_state_dict(self, path: str = None, train: bool = True):

        print('... loading state dict')
        checkpoint = torch.load(path)
        if train:
            start_epoch = checkpoint['epochs'] + 1
            self.d_opt.load_state_dict(checkpoint['d_opt'])
            self.g_opt.load_state_dict(checkpoint['g_opt'])
        else:
            start_epoch = 1
        
        self.d_A.module.load_state_dict(checkpoint['d_A'])
        self.d_B.module.load_state_dict(checkpoint['d_B'])
        self.g_A2B.module.load_state_dict(checkpoint['g_A2B'])
        self.g_B2A.module.load_state_dict(checkpoint['g_B2A'])
        
        return start_epoch
    
    
    @staticmethod
    def set_requires_grad(nets, requires_grad = False):
        
        """
        Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        
        if not isinstance(nets, list): nets = [nets]
        for net in nets:
            for param in net.module.parameters(): param.requires_grad = requires_grad
        
    
    def fit(self, nb_epochs: int = 200, d_lr: float = 2e-4, g_lr: float = 2e-4, beta_1: float = 0.5, model_name: \
            str = None, keep_only: int = 5, epoch_decay: int = 200, device=torch.device('cpu'), step_delay_fct=1):
        
        """
        Parameters: 
            model_name:  Resume the training from saved checkpoint "model_name"
            keep_only:   Max Number of models to keep in the self.save_dir
            epoch_decay: Number of epochs after which learning rate starts decaying
        """
        
        d_params = itertools.chain(self.d_A.module.parameters(),   self.d_B.module.parameters())
        g_params = itertools.chain(self.g_A2B.module.parameters(), self.g_B2A.module.parameters())
        
        self.d_opt = optim.Adam(params = d_params, lr = d_lr, betas = (beta_1, 0.999))
        self.g_opt = optim.Adam(params = g_params, lr = g_lr, betas = (beta_1, 0.999))
        
        start_epoch = 0; curr_iter = 0
        if model_name is not None: start_epoch = self.load_state_dict(path = self.save_dir + model_name)
        
        # LrScheduler follows this lambda rule to decay the learning rate
        def lr_lambda(epoch):
            fraction = (epoch - epoch_decay) / (nb_epochs - epoch_decay)
            return 1 if epoch < epoch_decay else 1 - fraction
        
        d_scheduler = optim.lr_scheduler.LambdaLR(self.d_opt, lr_lambda = lr_lambda, last_epoch = start_epoch - 1)
        g_scheduler = optim.lr_scheduler.LambdaLR(self.g_opt, lr_lambda = lr_lambda, last_epoch = start_epoch - 1)
        
        # TODO: NO GLOBAL SCRIPT CONSTANTS!!
        default_trafos = self.trn_dataloader.dataset.transforms
        if self.trn_dataloader is not None:
            trn_dataloader = self.trn_dataloader
        if device is not None:
            devices = [device]
        
        iter_count = 1
        for epoch in range(start_epoch + 1, nb_epochs + 1):
            p_bar = tqdm(total=int(len(trn_dataloader)))
            # if epoch <= nb_epochs /2 :
            #     trn_dataloader.dataset.transforms = T.Compose([default_trafos, TrafoWrapper(T.Resize([32, 32]))])
            # else:
            #     trn_dataloader.dataset.transforms = default_trafos
            self.fake_pool_A.empty()
            self.fake_pool_B.empty()
            for data in trn_dataloader:
                
                curr_iter += 1
                real_A, real_B = data['A'].to(devices[0]), data['B'].to(devices[0])
                
                # Forward pass
                fake_B = self.g_A2B(real_A)
                cyc_A = self.g_B2A(fake_B)

                fake_A = self.g_B2A(real_B)
                cyc_B = self.g_A2B(fake_A)

                idt_A  = self.g_B2A(real_A)
                real_B.cuda()
                idt_B = self.g_A2B(real_B)
                
                
                # No need to calculate the gradients for Discriminators' parameters
                self.set_requires_grad([self.d_A, self.d_B], requires_grad = False)
                d_A_pred_fake_data = self.d_A(fake_A)
                d_B_pred_fake_data = self.d_B(fake_B)
                
                # Cycle loss
                cyc_loss_A = self.loss.get_gen_cyc_loss(real_A, cyc_A)
                cyc_loss_B = self.loss.get_gen_cyc_loss(real_B, cyc_B)
                tot_cyc_loss = cyc_loss_A + cyc_loss_B
                
                # GAN loss
                g_A2B_gan_loss = self.loss.get_gen_gan_loss(d_B_pred_fake_data)
                g_B2A_gan_loss = self.loss.get_gen_gan_loss(d_A_pred_fake_data)
                
                # Identity loss
                g_B2A_idt_loss = self.loss.get_gen_idt_loss(real_A, idt_A)
                g_A2B_idt_loss = self.loss.get_gen_idt_loss(real_B, idt_B)
                
                # Total individual losses
                g_A2B_loss = g_A2B_gan_loss + g_A2B_idt_loss + tot_cyc_loss
                g_B2A_loss = g_B2A_gan_loss + g_B2A_idt_loss + tot_cyc_loss
                g_tot_loss = g_A2B_loss + g_B2A_loss - tot_cyc_loss
                
                # Parameters' getting updated
                self.g_opt.zero_grad()
                g_tot_loss.backward()
                if iter_count % step_delay_fct == 0:
                    self.g_opt.step()
                
                
                # Discriminator's optimization step
                self.set_requires_grad([self.d_A, self.d_B], requires_grad = True)
                self.d_opt.zero_grad()
                
                fake_A = self.fake_pool_A.push_and_pop(fake_A).to(devices[0])
                d_A_pred_real_data = self.d_A(real_A)
                d_A_pred_fake_data = self.d_A(fake_A.detach())
                
                # Discrimiator A loss
                d_A_loss = self.loss.get_dis_gan_loss(d_A_pred_real_data, d_A_pred_fake_data)
                d_A_loss.backward()
                
                fake_B = self.fake_pool_B.push_and_pop(fake_B).to(devices[0])
                d_B_pred_real_data = self.d_B(real_B)
                d_B_pred_fake_data = self.d_B(fake_B.detach())
                
                # Discrimiator B loss
                d_B_loss = self.loss.get_dis_gan_loss(d_B_pred_real_data, d_B_pred_fake_data)
                d_B_loss.backward() 
                
                # Parameters' getting updated
                if iter_count % step_delay_fct == 0:
                    self.d_opt.step()
                d_tot_loss = d_A_loss + d_B_loss
                
                
                if iter_count % step_delay_fct == 0:
                    p_bar.set_description(f'loss: {d_tot_loss.item():.3f}, {g_tot_loss.item():.3f}')
                p_bar.update(1)
                # # Writing statistics to the Tensorboard
                # self.tb.write_loss (d_tot_loss, g_tot_loss, epoch, curr_iter)
                # if curr_iter % 150 == 0: self.tb.write_image(10, self.g_A2B, self.g_B2A, epoch, curr_iter)
                iter_count += 1
            p_bar.close()
            
            
            curr_iter = 0
            g_scheduler.step()
            d_scheduler.step()
            
            print(f"After {epoch} epochs:"); 
            print(f"G_Loss: {round(g_tot_loss.item(), 3)}, D_Loss: {round(d_tot_loss.item(), 3)}", end = "\n")
            
            # Save models after every 10 epochs
            if epoch % 5 == 0:
                #self.saver.save_model(epoch, self.d_A, self.d_B, self.g_A2B, self.g_B2A, self.d_opt, self.g_opt)
                pass
    
    
    @torch.no_grad()
    def eval_(self, model_name: str = None, gen_images=False, save_path=None):

        print('... creating fake datasets')
        
        _ = self.load_state_dict(path = self.save_dir + model_name, train = False) 
        
        list_real_A = []
        list_fake_A = []
        list_real_B = []
        list_fake_B = []
        
        for idx, data in enumerate(val_dataloader):
            
            real_A = data['A'].to(devices[0])
            real_B = data['B'].to(devices[0])
            fake_A = self.g_B2A(real_B).detach()
            fake_B = self.g_A2B(real_A).detach()
            
            list_real_A.append(real_A)
            list_real_B.append(real_B)
            list_fake_A.append(fake_A)
            list_fake_B.append(fake_B)

            if gen_images:

                fake_A = fake_A.cpu().detach().numpy()/2 + 0.5
                fake_B = fake_B.cpu().detach().numpy()/2 + 0.5

                os.mkdir(root_dir+'fake_%s_val' %dataset_A)
                os.mkdir(root_dir+'fake_%s_val' %dataset_B)

                fig = plt.imshow(fake_A)
                plt.axis('off')
                fig.axes.get_xaxis().set_visible(False)
                fig.axes.get_yaxis().set_visible(False)
                plt.savefig(root_dir+'fake_%s_val/%s.jpg' %(dataset_A,str(idx)), bbox_inches='tight', pad_inches = 0)

                fig = plt.imshow(fake_B)
                plt.axis('off')
                fig.axes.get_xaxis().set_visible(False)
                fig.axes.get_yaxis().set_visible(False)
                plt.savefig(root_dir+'fake_%s_val/%s.jpg' %(dataset_B,str(idx)), bbox_inches='tight', pad_inches = 0)

        
        real_A = torch.cat(list_real_A, axis = 0)
        fake_A = torch.cat(list_fake_A, axis = 0)
        real_B = torch.cat(list_real_B, axis = 0)
        fake_B = torch.cat(list_fake_B, axis = 0)
        
        return real_A, real_B, fake_A, fake_B


def construct_cgan(root_dir = '', device=torch.device('cpu')):
    init = Initializer(init_type = 'normal', init_gain = 0.02)
    
    d_A = init(Discriminator(in_channels = 3, out_channels = 64, nb_layers = 3)).to(device)
    d_B = init(Discriminator(in_channels = 3, out_channels = 64, nb_layers = 3)).to(device)

    final_act = lambda x: torch.nn.functional.hardtanh(x, min_val=0.0, max_val=1.0)
    # g = lambda : nn.Sequential(
    #     #nn.Conv2d(3, 3, kernel_size=3, padding=1), nn.BatchNorm2d(3), nn.ReLU(),
    #     #nn.Conv2d(3, 3, kernel_size=1, padding=0), nn.BatchNorm2d(3), nn.ReLU(),
    #     nn.Conv2d(3, 3, kernel_size=1, padding=0), nn.Hardtanh(0.0, 1.0)
    # )
    # g_A2B = init(g())
    # g_B2A = init(g())
    g_A2B = init(Generic_UNet(input_channels=3, num_classes=3, num_pool=4, deep_supervision=False,
                              final_nonlin=final_act, base_num_features=30)).to(device)
    g_B2A = init(Generic_UNet(input_channels=3, num_classes=3, num_pool=4, deep_supervision=False,
                              final_nonlin=final_act, base_num_features=30)).to(device)
    # g_A2B = init(Generator(in_channels = 3, out_channels = 64, apply_dp = False)).to(device)
    # g_B2A = init(Generator(in_channels = 3, out_channels = 64, apply_dp = False)).to(device)
    
    model = CycleGAN(root_dir = root_dir, g_A2B = g_A2B, g_B2A = g_B2A, d_A = d_A, d_B = d_B)
    return model


def load_generators(dct_pth, gen_A=True, gen_B=False):
    model = construct_cgan()
    model.load_state_dict(dct_pth, train=False)

    mdls = []
    if gen_A:
        mdls.append(model.g_A2B)
    if gen_B:
        mdls.append(model.g_B2A)
    return mdls


def parser():
    ap = ArgumentParser()
    ap.add_argument('dataset', type=str)
    ap.add_argument('data', type=str)
    ap.add_argument('pretrained', type=str)
    ap.add_argument('--output_dir', type=str, default='outputs')
    ap.add_argument('--device', type=str, default='cuda')
    ap.add_argument('--model_B', action='store_true')
    return ap


# devices is global and needs to be set for this script --> dfault is CPU
devices = [torch.device('cpu')]


class ToSample:

    def __call__(self, img):
        return {'A': img, 'B': img}


class FromSample:

    def __call__(self, sample):
        return sample['A']


# main script
# main script
if __name__ == '__main__':
    torch.multiprocessing.set_sharing_strategy('file_system')

    hparams = parser().parse_args()
    hparams = main_patho_lightning.dset_shortcut(hparams)

    model_A = load_generators(hparams.pretrained, True, True)[0 if not hparams.model_B else 1]
    set_to_constructor = {
        'BreakHis': BreakHis,
        'IDCBreastHisto': IDCBreastHisto,
        'PCAM': PCAM,
    }
    img_sz = 96
    test_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize([img_sz, img_sz]),
        torchvision.transforms.ToTensor()
    ])
    if hparams.dataset == 'BreakHis':
        test_dataset = set_to_constructor[hparams.dataset](hparams.data, transform=test_transforms, split='test', multiclass=False)
    else:
        test_dataset = set_to_constructor[hparams.dataset](hparams.data, transform=test_transforms, split='test')
    img_shape = test_dataset[0][0].shape
    print(test_dataset.labels)
    test_dataset = torch.utils.data.DataLoader(test_dataset, batch_size=1, num_workers=4, shuffle=False)

    # fetching images
    device = torch.device(hparams.device)
    model_A = model_A.to(device)
    model_A.eval()
    lbls = []
    imgs = np.empty([len(test_dataset), *img_shape])
    origs = np.empty([len(test_dataset), *img_shape])
    with torch.no_grad():
        for idx, (img, lbl) in tqdm(enumerate(test_dataset), total=len(test_dataset)):
            new_img = model_A(img.to(device))
            imgs[idx] = new_img.detach().cpu().numpy()
            origs[idx] = img.detach().cpu().numpy()
            lbls.append(lbl.detach().cpu().numpy().astype(int))

    # writing imgs
    dir_robust = os.path.join(hparams.output_dir, 'modified')
    dir_orig = os.path.join(hparams.output_dir, 'original')
    if not os.path.isdir(dir_orig):
        os.makedirs(dir_orig)
    if not os.path.isdir(dir_robust):
        os.makedirs(dir_robust)
    for idx, ((img, orig), lbl) in tqdm(enumerate(zip(zip(imgs, origs), lbls)), total=len(test_dataset)):
        torchvision.utils.save_image(torch.from_numpy(img), os.path.join(dir_robust, f'{idx}_{lbl}.png'))
        torchvision.utils.save_image(torch.from_numpy(orig), os.path.join(dir_orig, f'{idx}_{lbl}.png'))

