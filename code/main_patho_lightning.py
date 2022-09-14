###############
#generic
import torch
from torch import nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms
import torch.nn.functional as F
import torchmetrics.functional as F_metrics

import json
import torchvision
import os
import argparse
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import copy
from copy import deepcopy

#################
#specific
import torchvision.models as models

from functools import partial
from pathlib import Path
import pandas as pd
import numpy as np

import gashis
from utils.transformer import *
from utils.schedulers import *
import timm_models as timm_models2
import timm.models as timm_models

from pytorch_LRP.transformer import vit
from pytorch_LRP import vgg as rap_vgg
from pytorch_LRP import resnet as rap_resnet
from pytorch_LRP import layers as rap_layers
from utils.dataloader import *
from utils import bootstrap_utils
import hybrid
from sklearn.metrics import roc_auc_score, roc_curve, auc
import numpy as np
import math

from utils.pesg_auc import PESG_AUC, auc_loss
from utils.sam import SAM

from itertools import chain
import re


torch.multiprocessing.set_sharing_strategy('file_system')


# COLOR constants
OKBLUE = '\033[94m'
OKGREEN = '\033[92m'
ENDC = '\033[0m'


####################################################################################################
# AUX. FUNCTIONS
####################################################################################################


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


def get_paramgroups(model,hparams,resnet=True):
    #https://discuss.pytorch.org/t/implementing-differential-learning-rate-by-parameter-groups/32903
    if(resnet):
        pgs = [[model.conv1,model.bn1,model.layer1,model.layer2], [model.layer3,model.layer4],[model.fc]]
    else:#densenet
        pgs = [[model.features],[model.classifier]]
    pgs = [[p.parameters() for p in pg] for pg in pgs]
    lgs = [{"params":chain(*pg), "lr":hparams.lr*pow(hparams.discriminative_lr_factor,len(pgs)-1-i)} for i,pg in enumerate(pgs)]
    return lgs


def _freeze_bn_stats(model, freeze=True):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
            if freeze:
                m.eval()
            else:
                m.train()


def sanity_check(model, state_dict_pre):
    """
    Linear classifier should not change any weights other than the linear layer.
    This sanity check asserts nothing wrong happens (e.g., BN stats updated).
    """
    print("=> loading state dict for sanity check")
    state_dict = model.state_dict()

    for k in list(state_dict.keys()):
        # only ignore fc layer
        if 'fc.1.weight' in k or 'fc.1.bias' in k or 'classifier.1.weight' in k or 'classifier.1.bias' in k:
            continue
        assert ((state_dict[k].cpu() == state_dict_pre[k].cpu()).all()), \
            '{} is changed in linear classifier training.'.format(k)

    print("=> sanity check passed.")


#from https://github.com/facebookresearch/mixup-cifar10/blob/master/train.py
def mixup_data(x, y, alpha=1.0):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


####################################################################################################
# MAIN MODULE
####################################################################################################

class PathoLightning(pl.LightningModule):

    def __init__(self, hparams, custom_model):
        super(PathoLightning, self).__init__()

        self.model = custom_model
        self.is_resnet = isinstance(self.model, rap_resnet.ResNet)

        self.hparams.update(vars(hparams))
        self.lr = self.hparams.lr

        if(self.hparams.optimizer == "sam"):
            self.automatic_optimization = False

        # define number of output classes
        self.num_classes = hparams.num_classes

        if(self.hparams.auc_maximization):
             #prior class probabilities are just dummy values here
             self.criterion = auc_loss([1./self.num_classes]*self.num_classes)#num_classes
        else:
            #self.criterion = F.binary_cross_entropy_with_logits #F.cross_entropy
            self.criterion = nn.CrossEntropyLoss()#F.cross_entropy 
        # acc for model selection
        self.model_selection = False
        self.best_state_dict = None
        self.best_val_auc = 0
        self.setup(None)

    def forward(self, x):
        return self.model(x)

    def relprop(self, *args, **kwargs):
        return self.model.relprop(*args, **kwargs)

    def _step(self, data_batch, batch_idx, train):
        # mixup: In essence, mixup trains a neural network on convex combinations of pairs of examples and their labels.
        if train and self.hparams.mixup_alpha>0:
             images, target_a, target_b, lam = mixup_data(data_batch[0], data_batch[1],
                                                          alpha=self.hparams.mixup_alpha)
             output = model(images)
             loss = mixup_criterion(self.criterion, output, target_a, target_b, lam)
        # normal training
        else:
            preds = self.forward(data_batch[0])
            # Corss Entropy expects Long
            loss = self.criterion(preds, data_batch[1].long())

            if(self.hparams.optimizer == "sam" and train):
                ##alternatively via closure
                def closure():
                    #_freeze_bn_stats(self,freeze=True)
                    loss = self.criterion(self.forward(data_batch[0]),data_batch[1].long())
                    #_freeze_bn_stats(self,freeze=freeze_bn)
                    self.manual_backward(loss)
                    return loss
                self.manual_backward(loss)
                opt = self.optimizers()
                opt.step(closure=closure)
                opt.zero_grad()

        # logging and returning proper information, during training only the loss is needed (backprop, no evaluation)
        # self.log("train_loss" if train else "val_loss", loss, prog_bar=True)
        return loss if train else {'loss': loss, "preds": preds.detach(), "targs": data_batch[1]}

    def training_step(self, train_batch, batch_idx):
        # call eval on batch normalization
        if self.hparams.linear_eval:
            _freeze_bn_stats(self)
        # normal iteration step on mini batch
        return self._step(train_batch, batch_idx, train=True)

    def validation_step(self, val_batch, batch_idx, dataloader_idx=0):
        return self._step(val_batch, batch_idx, train=False)

    def test_step(self, val_batch, batch_idx):
        return self.validation_step(val_batch, batch_idx)

    def test_epoch_end(self, outputs):
        return self._eval_epoch_end(outputs)

    def validation_epoch_end(self, l_outputs):
        print('')
        print(f'\n{80*"-"}\n{OKBLUE}validation ended:{ENDC}\n')
        outs_val, outs_test = l_outputs
        print(f'{OKBLUE}VALIDATION SPLIT{ENDC}')
        results_val = self._eval_epoch_end(outs_val)
        print(f'{OKBLUE}TEST SPLIT{ENDC}')
        self._eval_epoch_end(outs_test)
        # model selection
        val_auc = results_val[-1]
        if self.model_selection and self.best_val_auc < val_auc:
            self.best_state_dict = deepcopy(self.state_dict())
            self.best_val_auc = val_auc
            print(f'{OKGREEN}new best model{ENDC}')

    def _eval_epoch_end(self, outputs):
        # (stochastic) AUC maximization specific logging
        if self.hparams.auc_maximization:
            print("a:", self.criterion.a.mean(), "b:", self.criterion.b.mean(),"alpha:", self.criterion.alpha.mean())
        # per dataset evaluation (training and validation set used to be submitted for validation)
        # gathering outputs (NOTE: softmax is applied as we use the Cross Entropy)
        preds_all = torch.cat([torch.softmax(x['preds'], dim=1) for x in outputs])
        pred_lbls_all = torch.argmax(preds_all, dim=1)
        targs_all = torch.cat([x['targs'] for x in outputs])
        # calculating evaluation metrics
        confusion_matrix, acc, recall, precision, F1, ra_score = self.__validation_measures(outputs, single_class=(self.hparams.num_classes<=2))
        # printing information on metrics
        print(f'TP, TN, FP, FN:\n{confusion_matrix.detach().cpu().numpy()}\n' +
              f'Accuracy:       {acc}\n'+
              f'Recall:         {recall}\n'+
              f'Precision:      {precision}\n' +
              f'F1-Score:       {F1}\n' +
              f'ROC AUC-Score:  {ra_score}\n' +
              f'loss:           {torch.stack([x["loss"] for x in outputs]).mean()}\n{80*"-"}')
        # logging information on metrics
        self.log("Accuracy", acc)
        # self.log('Recall', recall)
        # self.log('Precision', precision)
        # self.log('F1-Score', F1)
        self.log('ROC AUC - score', ra_score)
        return acc, recall, precision, F1, ra_score

    def __validation_measures(self, outputs, single_class=True):
        # gathering outputs (NOTE: softmax is applied as we use the Cross Entropy)
        preds_all = torch.cat([torch.softmax(x['preds'], dim=1) for x in outputs]).detach()
        targs_all = torch.cat([x['targs'] for x in outputs]).detach().long()
        pred_lbls_all = torch.argmax(preds_all, dim=1)
        # calculate metrics
        acc = F_metrics.accuracy(pred_lbls_all, targs_all)
        precision, recall = F_metrics.precision_recall(pred_lbls_all, targs_all, average='macro', num_classes=hparams.num_classes)
        F1 = 2 * ((precision*recall) / (precision+recall)) if (precision+recall) > 0 else 0
        # calculating ROC AUC score
        ra_score = F_metrics.auroc(preds_all, targs_all, num_classes=self.hparams.num_classes, average='macro')
        confusion_matrix = F_metrics.confusion_matrix(pred_lbls_all, targs_all, num_classes=hparams.num_classes)
        return confusion_matrix, acc, recall, precision, F1, ra_score

    def on_fit_start(self):
        if self.hparams.linear_eval:
            print("copying state dict before training for sanity check after training")   
            self.state_dict_pre = copy.deepcopy(self.state_dict().copy())

    def on_fit_end(self):
        if self.hparams.linear_eval:
            sanity_check(self, self.state_dict_pre)

    @staticmethod
    def image_size(hparams):
        # if we use inception v3, images aught to be at least 299x299
        if hparams.arch in ['gashis', 'inception_v3', 'gashisResNet']:
            t_shape = [299, 299]
        elif hparams.arch[:3].lower() == 'vit' or hparams.arch[:4].lower() == 'swin' or hparams.arch[:8].lower() == 'convnext':
            t_shape = [224, 224]
        elif hparams.arch in ['vit_C', 'gashisViTC']:
            t_shape = [128, 128]
        elif hparams.dataset == 'BreaKHis':
            if hparams.no_multiclass:
                t_shape = [96, 96] #[100, 100]
            else:
                t_shape = [100, 100]
        elif hparams.dataset in ['IDCBreastHisto', 'Robust']:
            t_shape = [96, 96] #[100, 100]
        # otherwise we'd like to predict on 96x96 patches (NOTE: x400 magnifacation is expected, like in PCam)
        else:
            t_shape = [96, 96]
        return t_shape

    @staticmethod
    def prepare_transformations(hparams):
        # setting up augmentation
        # data normalization if specified
        if hparams.imagenet_stats:
            normalize = transforms.Normalize(mean=imagenet_stats[0], std=imagenet_stats[1])
        elif hparams.norm_data:
            norm_stats = None
            if hparams.dataset == 'PCam':
                norm_stats = PCam.NORM_STATS
            elif hparams.dataset == 'MHIST':
                norm_stats = MHIST.NORM_STATS
            elif hparams.dataset == 'BreaKHis':
                norm_stats = BreaKHis.NORM_STATS
            elif hparams.dataset == 'J_Nuclei':
                norm_stats = JanowczykNucleiClassification.NORM_STATS
            normalize = transforms.Normalize(*norm_stats) if norm_stats is not None else lambda x: x
        else:
            normalize = lambda x: x
        # these are the "Comparison of Deep Learning Approaches... transformations
        train_transforms = transforms.Compose([
                #transforms.RandomRotation(180),
                #transforms.RandomHorizontalFlip(),
                #transforms.RandomVerticalFlip(),
                #transforms.ColorJitter(hue=0.5),
                #transforms.ToTensor(),
                normalize,
            ])
        test_transforms = transforms.Compose([
            torch.nn.Identity() if not hparams.transform_test else transforms.ColorJitter(0.5),
            #transforms.ToTensor(),
            normalize,
        ])
        # 2 class BreaKHis and IDC have 96x96 input from CycleGAN
        pre_resize = torch.nn.Identity() if not (hparams.dataset in ['BreaKHis', 'IDCBreastHisto', 'Robust'] and hparams.num_classes == 2) else transforms.Resize([96, 96])
        t_shape = PathoLightning.image_size(hparams)
        train_transforms = transforms.Compose([pre_resize, train_transforms, transforms.Resize(t_shape), transforms.ToTensor()])
        test_transforms = transforms.Compose([pre_resize, test_transforms, transforms.Resize(t_shape), transforms.ToTensor()])
        return train_transforms, test_transforms

    def setup(self, stage):
        train_transforms, test_transforms = PathoLightning.prepare_transformations(self.hparams)

        print("Setting up data...")
        print('multiclass', not self.hparams.no_multiclass)

        self.train_dataset = PathoLightning.get_dataset(self.hparams, split='train', trafos=train_transforms)
        self.test_dataset = PathoLightning.get_dataset(self.hparams, split='test', trafos=test_transforms)
        # load a validataion_set, if present
        try:
            self.val_dataset = PathoLightning.get_dataset(self.hparams, split='valid', trafos=test_transforms)
            self.model_selection = True
        except:
            self.val_dataset = self.test_dataset
            self.model_selection = False
        print(f"Done. Labels: {self.train_dataset.labels}")
        if self.hparams.auc_maximization: #assign proper prior class probabilities
            lbl_encs = list(map(lambda i: (np.arange(self.hparams.num_classes) == i).astype(float), self.train_dataset.df.label.values))
            self.criterion.ps = torch.from_numpy(np.mean(lbl_encs, axis=0))

    @staticmethod
    def get_dataset(hparams, split, trafos=nn.Identity()):
        set_to_constructor = {
            'BreaKHis': lambda *args, **kwargs: BreaKHis(*args, **kwargs, multiclass=not hparams.no_multiclass),
            'PCam': PCam,
            'IDCBreastHisto': IDCBreastHisto,
            'MHIST': MHIST,
            'J_Nuclei': JanowczykNucleiClassification,
            'GasHisSDB': GasHisSDB,
            'Robust': RobustnessSampleSet
        }
        return set_to_constructor[hparams.dataset](hparams.data, transform=trafos, split=split,
                                                   rotate_90=hparams.rotate_90 if split=='train' else False,
                                                   ratios=hparams.ratios)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.hparams.batch_size, num_workers=4,
                          shuffle=True, drop_last=True, collate_fn=self.train_dataset.collate)

    def val_dataloader(self):
        #return DataLoader(self.val_dataset, batch_size=self.hparams.batch_size, num_workers=4)
        return [DataLoader(self.val_dataset, batch_size=self.hparams.batch_size, num_workers=4, collate_fn=self.val_dataset.collate),
                DataLoader(self.test_dataset, batch_size=self.hparams.batch_size, num_workers=4, collate_fn=self.test_dataset.collate)]

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.hparams.batch_size, num_workers=4, collate_fn=self.test_dataset.collate)

    def configure_optimizers(self):
        if hparams.auc_maximization:
            if self.hparams.linear_eval or self.hparams.train_head_only:
                params = [
                    {"params": (self.model.fc[-1].parameters() if len(self.hparams.lin_ftrs_head) > 0 else self.model.fc.parameters()) if self.is_resnet
                               else (self.model.classifier[-1].parameters() if len(self.hparams.lin_ftrs_head) > 0 else self.model.classifier.parameters()),
                    "lr": self.lr},
                    {"params": iter([self.criterion.a, self.criterion.b]), "lr": 100*self.lr},
                    {"params": iter([self.criterion.alpha]), "lr":100*self.lr, "is_alpha":True}
                ]
            else:
                params = get_paramgroups(
                    self.model,
                    self.hparams,
                    resnet=self.is_resnet
                ) + [
                    {"params": iter([self.criterion.a, self.criterion.b]), "lr":100*self.lr},
                    {"params": iter([self.criterion.alpha]), "lr":100*self.lr, "is_alpha":True}
                ]
            opt = PESG_AUC

        else:
            if self.hparams.optimizer == "sgd":
                opt = torch.optim.SGD
            elif self.hparams.optimizer == "adam":
                opt = torch.optim.AdamW
            elif(self.hparams.optimizer == "sam"):
                pass
            else:
                raise NotImplementedError("Unknown Optimizer.")

            if self.hparams.linear_eval or self.hparams.train_head_only:
                params = self.model.fc.parameters() if self.is_resnet else self.model.classifier.parameters()
            elif self.hparams.discriminative_lr_factor != 1.:#discrimative lrs
                params = get_paramgroups(self.model,self.hparams,resnet=self.is_resnet)
            else:
                params = self.parameters()

        if(self.hparams.optimizer=="sam"):
            base_optimizer = torch.optim.SGD  # define an optimizer for the "sharpness-aware" update
            # NOTE: weight decay potentially harmful; typically requires larger lrs
            optimizer = SAM(params, base_optimizer, lr=self.lr, momentum=0.9, rho=0.05, adaptive=True)#weight_decay=self.hparams.weight_decay
        else:
            if self.hparams.optimizer == "sgd":
                optimizer = opt(params, self.lr, weight_decay=self.hparams.weight_decay, momentum=0.9)
            else:
                optimizer = opt(params, self.lr, weight_decay=self.hparams.weight_decay)

        if(self.hparams.lr_schedule=="const"):
            n_schedule_epochs = 4
            scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1/n_schedule_epochs, end_factor=1.0,
                                                          total_iters=n_schedule_epochs)
            #scheduler = get_constant_schedule(optimizer)
        elif(self.hparams.lr_schedule=="warmup-const"):
            scheduler = get_constant_schedule_with_warmup(optimizer,self.hparams.lr_num_warmup_steps)
        elif(self.hparams.lr_schedule=="warmup-cos"):
            scheduler = get_cosine_schedule_with_warmup(optimizer,self.hparams.lr_num_warmup_steps,self.hparams.epochs*len(self.train_dataloader()),num_cycles=0.5)
        elif(self.hparams.lr_schedule=="warmup-cos-restart"):
            scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(optimizer,self.hparams.lr_num_warmup_steps,self.hparams.epochs*len(self.train_dataloader()),num_cycles=self.hparams.epochs-1)
        elif(self.hparams.lr_schedule=="warmup-poly"):
            scheduler = get_polynomial_decay_schedule_with_warmup(optimizer,self.hparams.lr_num_warmup_steps,self.hparams.epochs*len(self.train_dataloader()),num_cycles=self.hparams.epochs-1)   
        else:
            assert(False)

        return (
        [optimizer],
        [
            {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1,
            }
        ])

    def load_weights_from_checkpoint(self, checkpoint):
        """ Function that loads the weights from a given checkpoint file. 
        based on https://github.com/PyTorchLightning/pytorch-lightning/issues/525
        """
        checkpoint = torch.load(checkpoint, map_location=lambda storage, loc: storage,)
        if("state_dict" in checkpoint.keys()):#lightning style
            pretrained_dict = checkpoint["state_dict"]
        else:
            pretrained_dict = checkpoint
        model_dict = self.state_dict()
        missing_keys = [m for m in model_dict.keys() if not(m in pretrained_dict)]
        missing_keys_wo_num_batches_tracked = [m for m in missing_keys if not(m.endswith("num_batches_tracked"))]
        print(f"INFO:{len(model_dict)-len(missing_keys)} of {len(model_dict)} keys were matched.\n"
              f"{len(missing_keys_wo_num_batches_tracked)} missing keys (disregarding *.num_batches_tracked): {missing_keys_wo_num_batches_tracked}") 
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)


def resnet_arch(hparams):
    resnet_constructors = {
        #'resnet18': rap_resnet.resnet18,   #torchvision.models.resnet18,#
        #'resnet34': rap_resnet.resnet34,   #torchvision.models.resnet34,#
        'resnet50': timm_models2.resnetv2.resnetv2_50x1_bitm_in21k, #rap_resnet.resnet50,   #torchvision.models.resnet50,#
        #'resnet101': rap_resnet.resnet101, #torchvision.models.resnet101,#
        #'resnet152': timm_models.resnetv2.resnetv2_152, #torchvision.models.resnet152,#
        'resnet152x2': timm_models2.resnetv2.resnetv2_152x2_bitm_in21k, #torchvision.models.resnet152,#
        #'resnet152x4': timm_models.resnetv2.resnetv2_152x4_bitm_in21k, #torchvision.models.resnet152,#
    }
    resnet = resnet_constructors[hparams.arch](num_classes=hparams.num_classes, pretrained=hparams.use_pt_weights)
    # if isinstance(hparams.lin_ftrs_head, str):
    #     hparams.lin_ftrs_head = eval(hparams.lin_ftrs_head)
    # if not hparams.no_concat_pooling:
    #     resnet.avgpool = rap_layers.AdaptiveConcatPool2d()
    # resnet.fc = rap_create_head1d(2048, hparams.num_classes, lin_ftrs=hparams.lin_ftrs_head,
    #                              ps=hparams.dropout_head, concat_pooling=not hparams.no_concat_pooling,
    #                              bn=not hparams.no_bn_head)
    model = PathoLightning(hparams, resnet)
    return model


def vgg_arch(hparams):
    vgg_constructors = {
        'vgg11': rap_vgg.vgg11, 
        'vgg11_bn': rap_vgg.vgg11_bn, 
        'vgg13': rap_vgg.vgg13, 
        'vgg13_bn': rap_vgg.vgg13_bn, 
        'vgg16': rap_vgg.vgg16, 
        'vgg16_bn': rap_vgg.vgg16_bn,
        'vgg19_bn': rap_vgg.vgg19_bn, 
        'vgg19': rap_vgg.vgg19
    }
    return PathoLightning(hparams, vgg_constructors[hparams.arch](num_classes=hparams.num_classes))


def rap_bn_drop_lin(n_in, n_out, bn=True, p=0., actn=None):
    "Sequence of batchnorm (if `bn`), dropout (with `p`) and linear (`n_in`,`n_out`) layers followed by `actn`."
    layers = [rap_layers.BatchNorm1d(n_in)] if bn else []
    if p != 0: layers.append(rap_layers.Dropout(p))
    layers.append(rap_layers.Linear(n_in, n_out))
    if actn is not None: layers.append(actn)
    return layers
 

def rap_create_head1d(nf, nc, lin_ftrs=None, ps=0.5, bn_final:bool=False, bn:bool=True, act="relu", concat_pooling=True):
    "Model head that takes `nf` features, runs through `lin_ftrs`, and about `nc` classes; added bn and act here"
    lin_ftrs = [2*nf if concat_pooling else nf, nc] if lin_ftrs is None else [2*nf if concat_pooling else nf] + lin_ftrs + [nc] #was [nf, 512,nc]
    ps = [ps] if not(isinstance(ps,list)) else ps
    if len(ps)==1:
        ps = [ps[0]/2] * (len(lin_ftrs)-2) + ps
    actns = [rap_layers.ReLU(inplace=True)] * (len(lin_ftrs)-2) + [None]
    layers = []
    for ni,no,p,actn in zip(lin_ftrs[:-1],lin_ftrs[1:],ps,actns):
        layers += rap_bn_drop_lin(ni,no,bn,p,actn)
    if bn_final: 
        layers.append(rap_layers.BatchNorm1d(lin_ftrs[-1], momentum=0.01))
    return rap_layers.Sequential(*layers)


#####################################################################################################
# ARGPARSERS
#####################################################################################################
def add_model_specific_args(parser):
    parser.add_argument('--arch',default="densenet121", type=str,help='torchvision architecture- presently only resnets and densenets')
    parser.add_argument("--train-head-only", action="store_true", help="freeze everything except classification head (note: --linear-eval defaults to no hidden layer in classification head)")
    # Classifier args
    parser.add_argument("--dropout-head", type=float, default=0.5)
    parser.add_argument('--no-bn-head', action='store_true', help="use no batch normalization in classification head")
    parser.add_argument('--no-concat-pooling', action='store_true', help="use no concat pooling and standard mean pooling instead (applies to resnets only)")
    parser.add_argument('--lin-ftrs-head', type=str, default="[512]", help='linear filters for head (as string e.g. [1024] or [] for no extra hidden layers)')
    # ViT args
    parser.add_argument("--tokenizer-type", type=str, default="vit", help="vit/conv")
    parser.add_argument("--patch-size", type=int, default=4)
    parser.add_argument("--dim-model", type=int, default=768)
    parser.add_argument("--mlp-ratio", type=float, default=4.0)
    parser.add_argument("--n-heads", type=int, default=12)
    parser.add_argument("--n-layers", type=int, default=12)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--dropout-attention", type=float, default=0.1)
    parser.add_argument("--stochastic-depth-rate", type=float, default=0.1)
    parser.add_argument("--pos-enc", type=str, default="sine",help="none/sine/learned")
    parser.add_argument("--pooling-type", type=str, default="meanmax",help="cls/meanmax/meanmax-cls/seq/seq-meanmax/seq-meanmax-cls")
    parser.add_argument("--cls-token", action="store_true")
    parser.add_argument("--n-heads-seq-pool", type=int, default=1)
    # ViT_C args
    parser.add_argument('--stem-n-layers', type=int, default=5, help='number of 3x3 conv layers in the ViT_C stem')
    parser.add_argument('--stem-stride', type=int, default=2, help='stride length of the ViT_C stem, NOTE: only strie-1 and stride-2 convolutions are allowed, for stride-2 convolutions feature maps will be doubled at each layer.')
    parser.add_argument('--channels-in', type=int, default=3, help='number of (color-) channels of the input')
    parser.add_argument('--stem-skip-connections', action='store_true', help='will include skip connections and average pooling')
    return parser


def add_default_args():
    parser = argparse.ArgumentParser(description='PyTorch Lightning Patho Training')
    parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
    parser.add_argument("--dataset",type=str,help="chexpert/chexpert14/mimic_cxr/cxr14/chexphoto14/chexphoto", default="mimic_cxr")
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--batch-size', default=64, type=int,
                        metavar='N',
                        help='mini-batch size')
    parser.add_argument('--lr', '--learning-rate', default=0.0015, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--wd', '--weight-decay', default=1e-3, type=float,
                        metavar='W', help='weight decay (default: 0.)',
                        dest='weight_decay')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

    parser.add_argument('--pretrained', default='', type=str, metavar='PATH',
                    help='path to pretrained checkpoint (default: none)')
    parser.add_argument('--use-pt-weights', action='store_true')
    parser.add_argument('--optimizer', default='adam', help='sgd/adam/sam')
    parser.add_argument('--output-path', default='.', type=str,dest="output_path",
                        help='output path')
    parser.add_argument('--code-path', default='/opt/submit/', type=str, dest="code_path", help='code path')                   
    parser.add_argument('--metadata', default='', type=str,
                        help='metadata for output')
    
    parser.add_argument("--gpus", type=int, default=1, help="number of gpus")
    parser.add_argument("--num-nodes", dest="num_nodes", type=int, default=1, help="number of compute nodes")
    parser.add_argument("--precision", type=int, default=32, help="16/32")
    parser.add_argument("--distributed-backend", dest="distributed_backend", type=str, default=None, help="None/ddp")
    parser.add_argument("--accumulate", type=int, default=1, help="accumulate grad batches (total-bs=accumulate-batches*bs)")
        
    parser.add_argument("--linear-eval", action="store_true", help="linear evaluation instead of full finetuning",  default=False )

    parser.add_argument('--imagenet-stats', action='store_true', help='use imagenet stats instead of dataset stats (use with pretrained imagenet models)')
    parser.add_argument('--image-size', default=224, type=int, help='image size in pixels')
    parser.add_argument('--add-validation-set', action='store_true', help='split off validation set')
    parser.add_argument('--create-splits', action='store_true', help='option to create splits in a multi-process environment')
    parser.add_argument('--custom-split', action='store_true', help='custom stratified split 80:10:10')
    parser.add_argument('--random-split', action='store_true', help='random split diregarding patients 80:10:10')
    parser.add_argument('--training-fraction', default=1.0, type=float,dest="training_fraction", help='fraction of training examples to be used.')
    #parser.add_argument('--bootstrap-samples', default=0, type=int, dest='bootstrap_samples', help='number of bootstrap samples during evaluation (0 for no bootstrap)')
        
    parser.add_argument('--mixup-alpha', type=float, default=0.0, help='alpha>0 use mixup during training (default choice alpha=1)')
    parser.add_argument("--discriminative-lr-factor", type=float, help="factor by which the lr decreases per layer group during finetuning", default=1)#0.1)
    
    
    parser.add_argument("--lr-schedule", type=str, help="const/warmup-const/warmup-cos/warmup-cos-restart/warmup-poly", default="warmup-cos")
    parser.add_argument("--lr-num-warmup-steps", type=int, help="number of linear lr warmup steps", default=1000)
    parser.add_argument("--lr-find", action="store_true",  help="run lr finder before training run", default=False )
    
    parser.add_argument("--auc-maximization", action="store_true", help="direct auc maximization",  default=False )
    parser.add_argument('--refresh-rate', default=0, type=int, help='progress bar refresh rate (0 for disabled)')

    parser.add_argument('--rotate-90', default=False, type=bool, help='All samples will be also rotated by 0, 90, 180, 270 degr. Warning: batch will be four times as large.')
    
    # parameter on how often to run the training
    parser.add_argument('--k_experiments', default=1, type=int, help='number of experiment repetitions')
    parser.add_argument('--test-only', action="store_true", default=False, help='number of experiment repetitions')
    parser.add_argument('--num-classes', type=int, default=2, help='number of classes to predict')
    parser.add_argument('--norm-data', action='store_true', help='if this flag is set color channels will be normalized')
    parser.add_argument('--transform-test', action='store_true', help='if this flag is set, the test data will be augmented (robustness check)')
    parser.add_argument('--no-multiclass', action='store_true', help='no multiclass for BreaKHis')
    parser.add_argument('--ratios', default=None, type=float, nargs='+', help='if set, the class ratios will be set accordingly via oversampling')
    return parser


def transformer_input_size_sanity(hparams):
    # specifying input size here prevents errors when executing scripts for different datasets
    hparams.input_size = hparams.image_size
    if 'PCam' in hparams.dataset:
        hparams.input_size = 96
    elif 'BreaKHis' in hparams.dataset:
        hparams.input_size = 128
    if hparams.arch in ['vit', 'gashisViT'] or 'MHIST' in hparams.dataset:
        hparams.input_size = 224
    elif hparams.arch in ['vit_C', 'gashisViTC']:
        hparams.input_size = 128
    elif hparams.arch in ['gashis', 'inception_v3']:
        hparams.input_size = 299
    # input-dim of ViT
    hparams.dim_model = hparams.patch_size**2 * 3
    return hparams


def construct_model(hparams, **kwargs):
    if hparams.arch in [f'resnet{n}' for n in [18, 34, 50, 101, 152, '152x2', '152x4']]:
        model = resnet_arch(hparams)
    elif hparams.arch in ['vit', 'vit-B']:
        model = PathoLightning(
            hparams,
            timm_models.vit_base_patch16_224_in21k(pretrained=hparams.use_pt_weights, num_classes=hparams.num_classes)
        )
        #model = PathoLightning(
        #    hparams, 
        #    vit.VisionTransformer(
        #        img_size=hparams.input_size, patch_size=hparams.patch_size, in_chans=3, num_classes=hparams.num_classes,
        #        embed_dim=hparams.dim_model, depth=hparams.n_layers, num_heads=hparams.n_heads, 
        #        mlp_ratio=hparams.mlp_ratio, drop_rate=hparams.dropout, attn_drop_rate=hparams.dropout_attention
        #    )
        #)
    elif hparams.arch == 'vit-S':
        model = PathoLightning(
            hparams,
            timm_models.vit_small_patch16_224_in21k(pretrained=hparams.use_pt_weights, num_classes=hparams.num_classes)
        )
    elif hparams.arch == 'vit-L':
        model = PathoLightning(
            hparams,
            timm_models2.vit_large_patch16_224_in21k(pretrained=hparams.use_pt_weights, num_classes=hparams.num_classes)
        )
    elif hparams.arch == 'vit-T':
        model = PathoLightning(
            hparams,
            timm_models2.vit_tiny_patch16_224_in21k(pretrained=hparams.use_pt_weights, num_classes=hparams.num_classes)
        )
    elif hparams.arch == 'vit-H':
        model = PathoLightning(
            hparams,
            timm_models.vit_huge_patch14_224_in21k(pretrained=hparams.use_pt_weights, num_classes=hparams.num_classes)
        )
    elif hparams.arch == 'swin-T':
        model = PathoLightning(
            hparams,
            timm_models.swin_tiny_patch4_window7_224(pretrained=hparams.use_pt_weights, num_classes=hparams.num_classes)
        )
    elif hparams.arch == 'swin-B':
        model = PathoLightning(
            hparams,
            timm_models.swin_base_patch4_window7_224_in22k(pretrained=hparams.use_pt_weights, num_classes=hparams.num_classes)
        )
    elif hparams.arch == 'swin-L':
        model = PathoLightning(
            hparams,
            timm_models.swin_large_patch4_window7_224_in22k(pretrained=hparams.use_pt_weights, num_classes=hparams.num_classes)
        )
    elif hparams.arch == 'swin-S':
        model = PathoLightning(
            hparams,
            timm_models.swin_small_patch4_window7_224(pretrained=hparams.use_pt_weights, num_classes=hparams.num_classes)
        )
    elif hparams.arch == 'convnext-T':
        model = PathoLightning(
            hparams,
            timm_models2.convnext_tiny_in22k(pretrained=hparams.use_pt_weights, num_classes=hparams.num_classes, 
                                             drop_rate=0.1, drop_path_rate=0.5)
        )
    elif hparams.arch == 'convnext-S':
        model = PathoLightning(
            hparams,
            timm_models.convnext_small_in22k(pretrained=hparams.use_pt_weights, num_classes=hparams.num_classes)
        )
    elif hparams.arch == 'convnext-B':
        model = PathoLightning(
            hparams,
            timm_models.convnext_base_in22k(pretrained=hparams.use_pt_weights, num_classes=hparams.num_classes)
        )
    elif hparams.arch == 'convnext-L':
        model = PathoLightning(
            hparams,
            timm_models2.convnext_large_in22k(pretrained=hparams.use_pt_weights, num_classes=hparams.num_classes)
        )
    elif hparams.arch == 'convnext-XL':
        model = PathoLightning(
            hparams,
            timm_models.convnext_xlarge_in22k(pretrained=hparams.use_pt_weights, num_classes=hparams.num_classes)
        )
    elif hparams.arch == 'ViTCnXt-T':
        model = PathoLightning(
            hparams,
            hybrid.ViTT_ConvNeXtT(hparams)
        )
    elif hparams.arch == 'ViTCnXt-L':
        model = PathoLightning(
            hparams,
            hybrid.ViTL_ConvNeXtL(hparams)
        )
    elif hparams.arch == 'SwinCnXt-T':
        model = PathoLightning(
            hparams,
            hybrid.SwinT_ConvNeXtT(hparams)
        )
    elif hparams.arch == 'SwinCnXt-L':
        model = PathoLightning(
            hparams,
            hybrid.SwinL_ConvNeXtL(hparams)
        )
    elif hparams.arch == 'IV3CnXt-T':
        model = PathoLightning(
            hparams,
            hybrid.ConvNeXtT_InceptionV3(hparams)
        )
    elif hparams.arch == 'vit_C':
        model = PathoLightning(hparams, vit.VisionTransformerConvStem(hparams, **kwargs))
    elif hparams.arch == 'gashis':
        model = PathoLightning(hparams, gashis.GasHisTransformer(hparams))
    elif hparams.arch == 'gashisResNet':
        model = PathoLightning(hparams, gashis.ResNet50InceptionV3(hparams))
    elif hparams.arch == 'gashisViT':
        model = PathoLightning(hparams, gashis.ViTInceptionV3(hparams))
    elif hparams.arch == 'gashisViTC':
        model = PathoLightning(hparams, gashis.ViTCInceptionV3(hparams, **kwargs))
    elif hparams.arch == 'botnet50':
        print(f'siz_in={PathoLightning.image_size(hparams)[0]}')
        model = PathoLightning(hparams, gashis.BotNet50(n_classes=hparams.num_classes, size_in=PathoLightning.image_size(hparams)[0], pretrained=hparams.use_pt_weights))
    elif hparams.arch == 'inception_v3':
        model = PathoLightning(hparams, timm_models2.inception_v3(pretrained=hparams.use_pt_weights, num_classes=hparams.num_classes))#gashis.Inception(n_classes=hparams.num_classes))
    elif hparams.arch.startswith('vgg'):
        model = vgg_arch(hparams)
    else:
        model = PathoLightning(hparams)
    # pretrained
    if hparams.pretrained != '':
        model.load_weights_from_checkpoint(hparams.pretrained)
    return model


def construct_logger(hparams):
    logger = TensorBoardLogger(
        save_dir=hparams.output_path,
        name="model_logs"
    )
    print("Output directory:", logger.log_dir)
    return logger


def construct_trainer(hparams):
    trainer = pl.Trainer(
        auto_lr_find=hparams.lr_find,
        accumulate_grad_batches=hparams.accumulate,
        max_epochs=hparams.epochs,
        min_epochs=hparams.epochs,

        default_root_dir=hparams.output_path,

        num_sanity_val_steps=0,

        logger=logger,
        benchmark=True,

        gpus=hparams.gpus,
        num_nodes=hparams.num_nodes,
        precision=hparams.precision,

        progress_bar_refresh_rate=hparams.refresh_rate,
        weights_summary='top'
    )
    return trainer


###################################################################################################
# MAIN
###################################################################################################


def parser():
    ap = add_default_args()
    ap = add_model_specific_args(ap)
    return ap


def dset_shortcut(hparams):
    dset_path_dict_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'meta_data', 'dset_paths.json')
    if os.path.isfile(dset_path_dict_file):
        with open(dset_path_dict_file, 'r') as f_json:
            dset_path_dict = json.load(f_json)
            if hparams.data in dset_path_dict.keys():
                hparams.data = dset_path_dict[hparams.data]
    return hparams


if __name__ == '__main__':
    ap = parser()
    hparams = ap.parse_args()
    hparams.executable = "main_lincls_lightning"

    if not os.path.exists(hparams.output_path):
        os.makedirs(hparams.output_path)

    # checking if there is dset_paths.json in key2ai-med/meta_data/, if so dataset-paths can just be read
    hparams = dset_shortcut(hparams)

    # automaticly setting the input size for transformers
    hparams = transformer_input_size_sanity(hparams)

    # repeating experiments k times (allows for mean and std of AUC/ Accuracy, reported numbers are solid)
    results = []
    model, logger, trainer = None, None, None
    pretrained = hparams.pretrained
    for idx_experiment in range(hparams.k_experiments):
        # GPU  sanity for k_experiments > 1
        del model
        del logger 
        del trainer
        torch.cuda.empty_cache()

        # loading pretrained model, if specified           
        if pretrained != "":
            if not os.path.isfile(pretrained):
                if os.path.isdir(pretrained):
                    hparams.pretrained = sorted([os.path.join(pretrained, p) for p in os.listdir(pretrained) if p.endswith('.pth')])[idx_experiment]
                else:
                    raise FileNotFoundError(f'{hparams.pretrained} is neither file nor directory.')
        # Model initialization
        model = construct_model(hparams)

        # logging / stdout config
        logger = construct_logger(hparams)

        # training and testing
        trainer = construct_trainer(hparams)

        #lr find
        if(hparams.lr_find):
            trainer.tune(model)

        # training of model
        # NOTE: ckpt_path=None implies the usage of the latest weights after the final optimization step.
        #       No model selection based on validation set is performed, as the test set may pose as the validation set.
        if not hparams.test_only:
            trainer.fit(model)
            # model selection if possible
            if model.model_selection and model.best_state_dict is not None:
                model.load_state_dict(model.best_state_dict)
            # testing of model
            res = trainer.test(model, model.test_dataloader(), ckpt_path=None)[0]#(ckpt_path=None)[0]
            results.append(res)
            str_leading_0 = "0" * (len(str(hparams.k_experiments)) - len(str(idx_experiment)))
            if not os.path.isdir(os.path.join(hparams.output_path, 'models')):
                os.makedirs(os.path.join(hparams.output_path, 'models'))
            # logginh/ saving the models state
            torch.save(model.state_dict(),
                       os.path.join(hparams.output_path, 'models', f'experiment_{str_leading_0}{idx_experiment}.pth'))
            # logging/ saving the models hyper parameters as a json
            with open(os.path.join(hparams.output_path, 'models', f'experiment_{str_leading_0}{idx_experiment}.json'),
                      'w') as f_hparams:
                json.dump(vars(hparams), f_hparams)
            print(f'{OKGREEN}model written to {os.path.join(hparams.output_path, "models", f"experiment_{str_leading_0}{idx_experiment}.pth")}{ENDC}')
        else:
            res = trainer.test(model, model.test_dataloader(), ckpt_path=None)[0]
            results.append(res)

    # empirical bootstrap
    print(f'{OKGREEN}boot strap eval:{ENDC}')
    def F_acc(Y_hat, Y): return F_metrics.accuracy(torch.argmax(Y_hat, dim=1), Y)
    def F_auroc(Y_hat, Y): return F_metrics.auroc(torch.softmax(Y_hat, dim=1), Y, num_classes=hparams.num_classes, average='macro')
    metrices_dict = {
        'accuracy': F_acc,
        'ROC': F_auroc
    }
    bootstrap_results = {}
    for name, score_fn in metrices_dict.items():
        empircical_results = bootstrap_utils.bootstrap_model_eval(model, model.test_dataloader(), score_fn, threads=None)[:3]
        bootstrap_results[name] = empircical_results
    print(bootstrap_results)

    # saving empirical bootstrap results
    with open(os.path.join(hparams.output_path, 'BOOTSTRAP_RESULTS.json'), 'w') as f_report:
        json.dump(bootstrap_results, f_report)

    # calculating mean and standard deviation for all measures (e.g. accuracy and AUC) across the experiments
    print(f'{OKGREEN}experiments deviation eval:{ENDC}')
    report = {}
    for k in results[0].keys():
        all_vals = [r[k] for r in results]
        std = np.std(all_vals)
        mean = np.mean(all_vals)
        report[k] = {"mean": float(mean), "std": float(std)}
    print(report)

    # saving mean and std into a REPORT.json
    with open(os.path.join(hparams.output_path, 'REPORT.json'), 'w') as f_report:
        json.dump(report, f_report)

    # printing a tabular representation of the json file, rows look like <MEASURE>: <MEAN> | <STD>
    str_report_table = [f'{k}: {report[k]["mean"]} | {report[k]["std"]}' for k in report.keys()]
    str_report_table = '\n'.join(str_report_table)
    print(f'All experiments were completed\n{str_report_table}')

