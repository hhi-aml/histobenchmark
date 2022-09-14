import os
import numpy as np
import json
import pytorch_lightning as pl
import torch
from torch.nn import MSELoss, CrossEntropyLoss
from torch.utils.data import DataLoader
from torch import optim, nn
from torchvision.transforms import *
from nnunet.network_architecture.generic_UNet import Generic_UNet

from utils import dataloader
from main_patho_lightning import add_default_args, add_model_specific_args


def IoU(batch_imgs, batch_masks):
    assert batch_imgs.shape[0] == batch_masks.shape[0], 'batches aught to be of equal length'
    if isinstance(batch_imgs, np.ndarray):
        batch_imgs = torch.from_numpy(batch_imgs)
    if isinstance(batch_masks, np.ndarray):
        batch_masks = torch.from_numpy(batch_masks)
    sum_ = lambda b: b.type(torch.float32).sum(axis=1).sum(axis=1)
    batch_imgs = torch.squeeze(batch_imgs, dim=1)
    batch_masks = torch.squeeze(batch_masks, dim=1)
    b1 = batch_imgs > 0
    b2 = batch_masks > 0
    nom = sum_(b1*b2)
    denom = sum_(b1+b2)
    # avoid 0 div if there are no segmented areas in GT and prediction IoU is 1
    scores = torch.ones(nom.shape, dtype=torch.float32).to(b1.device)
    scores[denom>0] = 1
    scores[denom>0] = nom[denom>0] / denom[denom>0]
    return scores.cpu()


def args_parser():
    ap = add_model_specific_args(add_default_args())
    ap.add_argument('--augment', action='store_true', help='if this flag is set, data will be augmented during training')
    ap.add_argument('--img-shape', nargs='+', type=int, help='shape of the input (height, width)', default=(96, 96))
    ap.add_argument('--dset-stretch-factor', default=1, type=int,
                    help='Janowczyk data is scarce, hence prolonging an epoch/ delaying evaluations can prevent highly frequent logging.')
    return ap


class D2ToD3(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x[:,:,None,:,:]


class D3ToD2(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.mean(dim=2)


class LightningSegmentation(pl.LightningModule):

    def __init__(self, model, hparams, train, test, val=None):
        super().__init__()
        self.model = model
        self.train_dataset = train
        self.val_dataset = val if val is not None else test
        self.test_dataset = test
        self.hparams.update(vars(hparams))
        self.criterion = CrossEntropyLoss()

    def forward(self, x_in):
        return self.model(x_in)

    def training_step(self, batch, idx):
        return self._step(*batch, train=True)

    def validation_step(self, batch, idx):
        return self._step(*batch, train=False)

    def test_step(self, batch, idx):
        return self.validation_step(batch, idx)

    def _step(self, x_in, y, train):
        preds = self.forward(x_in)
        if not train:
            preds = nn.functional.log_softmax(preds, dim=1)
        loss = self.criterion(preds, y.long().squeeze())
        return loss if train else {'loss': loss.cpu().detach().numpy(), 'preds': preds.cpu().detach().numpy(), 'targs': y.cpu().detach().numpy()}

    def test_epoch_end(self, outs):
        self.validation_epoch_end(outs)

    def validation_epoch_end(self, outs):
        mean_loss = np.mean([o["loss"] for o in outs])
        targs = [o["targs"] for o in outs]
        preds = [(o['preds'][:,0,:,:] - o['preds'][:,1,:,:]) < 0 for o in outs]
        iou = 0
        N = 0
        for t, p in zip(targs, preds):
            t = torch.from_numpy(t).to(self.device)
            p = torch.from_numpy(p).to(self.device)
            iou += float(IoU(p, t).sum())
            N += t.shape[0]
        iou = iou / N
        print(80*'=')
        print(f'Validation Report\n\nLoss: {mean_loss}\nIoU:  {iou}')
        print(80*'=')

    def configure_optimizers(self):
        return optim.AdamW(self.parameters(), self.hparams.lr, weight_decay=self.hparams.weight_decay)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.hparams.batch_size, num_workers=4, shuffle=True, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.hparams.batch_size, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.hparams.batch_size, num_workers=1)


def gen_model(hparams, train, test, arch=None):
    if arch is None:
        arch = hparams.arch
    if arch == 'unet':
        unet = Generic_UNet(input_channels=3, base_num_features=30, num_classes=hparams.num_classes, num_pool=4, deep_supervision=False,
                            final_nonlin=lambda x: x)#torch.nn.LogSoftmax(dim=1))
        model = LightningSegmentation(unet, hparams, train, test)
    elif arch == 'unet3d':
        num_pool = 4
        unet = Generic_UNet(input_channels=3, base_num_features=30, num_classes=hparams.num_classes, num_pool=num_pool, deep_supervision=False,
                            final_nonlin=lambda x: x, #torch.nn.LogSoftmax(dim=1), 
                            conv_op=nn.Conv3d, norm_op=nn.BatchNorm3d, dropout_op=nn.Dropout3d,
                            pool_op_kernel_sizes=[(1, 2, 2)] * num_pool)
        unet = nn.Sequential(D2ToD3(), unet, D3ToD2())
        model = LightningSegmentation(unet, hparams, train, test)
    else:
        raise NotImplementedError(f'Unknown arch {hparams.arch} for segemntation, please choose from: [unet]')
    if hparams.pretrained != '':
        state_dict = torch.load(hparams.pretrained, map_location=model.device)
        if 'state_dict' in state_dict.keys():
            state_dict = state_dict['state_dict']
        model.load_state_dict(state_dict)
    return model


def gen_datasets(hparams):
    img_shape = hparams.img_shape
    img_trafos = [Resize(img_shape), ToTensor()]
    seg_trafos = Compose([Resize(img_shape), Grayscale(), ToTensor()])
    if hparams.dataset == 'J_Nuclei':
        if hparams.norm_data:
            img_trafos.append(Normalize(*dataloader.JanowczykNuclei.NORM_STATS))
        img_trafos = Compose(img_trafos)
        train = dataloader.JanowczykNuclei(hparams.data, split='train', img_transform=img_trafos, seg_mask_transform=seg_trafos, augment=hparams.augment)
        train.stretch(hparams.dset_stretch_factor)
        test = dataloader.JanowczykNuclei(hparams.data, split='test', img_transform=img_trafos, seg_mask_transform=seg_trafos)
    elif hparams.dataset == 'J_Epi':
        if hparams.norm_data:
            img_trafos.append(Normalize(*dataloader.JanowczykEpi.NORM_STATS))
        img_trafos = Compose(img_trafos)
        train = dataloader.JanowczykEpi(hparams.data, split='train', img_transform=img_trafos, seg_mask_transform=seg_trafos, augment=hparams.augment)
        train.stretch(hparams.dset_stretch_factor)
        test = dataloader.JanowczykEpi(hparams.data, split='test', img_transform=img_trafos, seg_mask_transform=seg_trafos)
    elif hparams.dataset == 'J_Tubule':
        if hparams.norm_data:
            img_trafos.append(Normalize(*dataloader.JanowczykTubule.NORM_STATS))
        img_trafos = Compose(img_trafos)
        train = dataloader.JanowczykTubule(hparams.data, split='train', img_transform=img_trafos, seg_mask_transform=seg_trafos, augment=hparams.augment)
        train.stretch(hparams.dset_stretch_factor)
        test = dataloader.JanowczykTubule(hparams.data, split='test', img_transform=img_trafos, seg_mask_transform=seg_trafos)
    elif hparams.dataset == 'DSBNuclei':
        if hparams.norm_data:
            img_trafos.append(Normalize(*dataloader.DataScienceBowlNuclei.NORM_STATS))
        img_trafos = Compose(img_trafos)
        train = dataloader.DataScienceBowlNuclei(hparams.data, split='train', img_transform=img_trafos, seg_mask_transform=seg_trafos, augment=hparams.augment)
        train.stretch(hparams.dset_stretch_factor)
        test = dataloader.DataScienceBowlNuclei(hparams.data, split='test', img_transform=img_trafos, seg_mask_transform=seg_trafos)
    elif hparams.dataset == 'PMoNuSeg':
        if hparams.norm_data:
            img_trafos.append(Normalize(*dataloader.PatchedMoNuSeg.NORM_STATS))
        img_trafos = Compose(img_trafos)
        train = dataloader.PatchedMoNuSeg(hparams.data, split='train', img_transform=img_trafos, seg_mask_transform=seg_trafos, augment=hparams.augment)
        train.stretch(hparams.dset_stretch_factor)
        test = dataloader.PatchedMoNuSeg(hparams.data, split='test', img_transform=img_trafos, seg_mask_transform=seg_trafos)
    else:
        raise NotImplementedError('Unknown dataset {hparams.dataset}, please choose from: [J_Nuclei, J_Epi, J_Tubule]')
    return train, test


if __name__ == '__main__':
    hparams = args_parser().parse_args()
    # checking if there is dset_paths.json in key2ai-med/meta_data/, if so dataset-paths can just be read
    dset_path_dict_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'meta_data', 'dset_paths.json')
    if os.path.isfile(dset_path_dict_file):
        with open(dset_path_dict_file, 'r') as f_json:
            dset_path_dict = json.load(f_json)
            if hparams.data in dset_path_dict.keys():
                hparams.dataset = hparams.data
                hparams.data = dset_path_dict[hparams.data]

    # data loading
    train, test = gen_datasets(hparams)
    # model initialization
    model = gen_model(hparams, train, test)
    # training and testing
    trainer = pl.Trainer(
        accumulate_grad_batches=hparams.accumulate,
        max_epochs=hparams.epochs,
        min_epochs=hparams.epochs,

        default_root_dir=hparams.output_path,
        num_sanity_val_steps=0,
        benchmark=True,

        gpus=hparams.gpus,
        num_nodes=hparams.num_nodes,
        precision=hparams.precision,
        distributed_backend=hparams.distributed_backend,
        progress_bar_refresh_rate=hparams.refresh_rate,
        weights_summary='top',
        resume_from_checkpoint= None if hparams.resume=="" else hparams.resume
    )
    if not hparams.test_only:
        trainer.fit(model)
        results = trainer.test(ckpt_path=None)
    else:
        results = trainer.test(model, ckpt_path=None)
    print(results)

