from argparse import ArgumentParser
import os
import sys
import torch
from torchvision.transforms import *
FILE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(FILE_DIR)
sys.path.append(os.path.join(FILE_DIR, '..'))
import cycle_gan


class CGANTrainer:

    def __init__(self, cgan, dset, dloader, hparams):
        self.cgan = cgan
        self.dset = dset
        self.dsetloader = dloader
        self.cgan.trn_dataloader = self.dsetloader
        self.hparams = hparams

    def fit(self):
        self.cgan.fit(nb_epochs=hparams.epochs, model_name=None , epoch_decay=hparams.epochs//2, 
                      device=torch.device(self.hparams.device), step_delay_fct=hparams.step_delay_fct)

    def save_model(self, other_path=''):
        pth = hparams.output_path if other_path == '' else other_path
        if not os.path.isdir(pth):
            os.makedirs(pth)
        pth = os.path.join(pth, 'cgan.pth')
        torch.save({'d_A': self.cgan.d_A.module.state_dict(), 
                    'd_B': self.cgan.d_B.module.state_dict(),
                    'g_A2B': self.cgan.g_A2B.module.state_dict(), 
                    'g_B2A': self.cgan.g_B2A.module.state_dict()}, pth)


def load_dsets(hparams, trafos):
    helper = cycle_gan.Helper()
    dset, dloader = helper.get_data(hparams.pathA, hparams.pathB, trafos, hparams.batch_size, is_train=True)
    return dset, dloader


def parser():
    ap = ArgumentParser()
    ap.add_argument('pathA', type=str)
    ap.add_argument('pathB', type=str)
    ap.add_argument('output_path', type=str)
    ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--device', type=str, default='cuda')
    ap.add_argument('--step-delay-fct', type=int, default=1)
    return ap


if __name__ == '__main__':
    hparams = parser().parse_args()
    cgan = cycle_gan.construct_cgan(device=torch.device(hparams.device))
    default_trafos = [
        Resize([96, 96]),
        RandomHorizontalFlip(p=0.5),
        RandomVerticalFlip(p=0.5),
        ToTensor()
    ]
    trafos = list(map(cycle_gan.TrafoWrapper, default_trafos))
    dset, dloader = load_dsets(hparams, trafos)
    trainer = CGANTrainer(cgan, dset, dloader, hparams)
    trainer.fit()
    trainer.save_model()

