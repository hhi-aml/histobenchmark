import torch
from torch.utils.data import DataLoader
from torchvision.transforms import functional as F_trafo
import torchmetrics.functional as F_metrics
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from collections import defaultdict
FILE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(FILE_DIR)
sys.path.append(os.path.join(FILE_DIR, '..'))
import main_patho_lightning


def hue_preds(model, dset, hue_range=[-0.5, 0.5], step=0.1, n_classes=2, batch_size=4, device=None):
    hue_range = hue_range + np.array([0, step]) # we want to cover the entire range (including last val)
    hue_range = np.arange(*hue_range, step=step)
    print(hue_range)
    d_loader = DataLoader(dset, batch_size=batch_size, num_workers=4)
    # storage of ground truth and predictions
    labels = np.zeros(len(dset))
    pred_labels = np.zeros([len(hue_range), len(dset), n_classes])
    # evaluating the model for every hue_factor
    with torch.no_grad():
        model.eval()
        if device is not None:
            model = model.to(device)
        for i, hue_factor in enumerate(hue_range):
            c = 0
            for imgs, gt in tqdm(d_loader):
                imgs = torch.stack([F_trafo.adjust_hue(img, hue_factor) for img in imgs])
                if device is not None:
                    imgs = imgs.to(device)
                pred_labels[i, c:c+len(gt)] = model(imgs).cpu().numpy()
                if i == 0:
                    labels[c:c+len(gt)] = gt.cpu().numpy()
                c += len(gt)
    return labels, pred_labels
    

def plot_curve(hue_vals, results_dct, out):
    for key in results_dct.keys():
        plt.plot(hue_vals, results_dct[key], label=key)
    plt.legend()
    if not os.path.isdir(os.path.dirname(out)):
        os.makedirs(os.path.dirname(out))
    plt.xticks(np.arange(hue_vals[0], hue_vals[-1]+hue_vals[-1]/10, step=abs(hue_vals[0]-hue_vals[-1])/10))
    sns.despine()
    plt.ylim()
    plt.savefig(os.path.join(out, 'plot.png'))


def validation_measures(targs_all, preds_all, n_classes=2):
    # gathering outputs (NOTE: softmax is applied as we use the Cross Entropy)
    preds_all = torch.softmax(torch.from_numpy(preds_all), dim=1)
    targs_all = torch.from_numpy(targs_all).long()
    pred_lbls_all = torch.argmax(preds_all, dim=1).long()
    # calculate metrics
    acc = F_metrics.accuracy(pred_lbls_all, targs_all)
    precision, recall = F_metrics.precision_recall(pred_lbls_all, targs_all, average='macro', num_classes=n_classes)
    F1 = 2 * ((precision*recall) / (precision+recall)) if (precision+recall) > 0 else 0
    # calculating ROC AUC score
    ra_score = F_metrics.auroc(preds_all, targs_all, num_classes=n_classes, average='macro')
    confusion_matrix = F_metrics.confusion_matrix(pred_lbls_all, targs_all, num_classes=n_classes)
    return confusion_matrix, acc, recall, precision, F1, ra_score


if __name__ == '__main__':
    hparams = main_patho_lightning.parser().parse_args()
    if not os.path.exists(hparams.output_path):
        os.makedirs(hparams.output_path)
    hparams = main_patho_lightning.dset_shortcut(hparams)
    hparams = main_patho_lightning.transformer_input_size_sanity(hparams)
    # loading the model and dataset
    model = main_patho_lightning.construct_model(hparams)
    dset = model.test_dataloader().dataset
    # fixing hue value range
    HUE_VALS = (-0.5, 0.5)
    HUE_STEP = 0.05
    # estimating labels
    labels, pred_labels = hue_preds(model, dset, HUE_VALS, HUE_STEP, device=torch.device('cuda'), batch_size=hparams.batch_size)
    # calculating metrices
    results_dct = defaultdict(list)
    for preds in pred_labels:
        confusion_matrix, acc, recall, precision, F1, ra_score = validation_measures(labels, preds)
        results_dct['Acc'].append(acc)
        results_dct['Prec'].append(precision)
        results_dct['Rec'].append(recall)
        results_dct['F1'].append(F1)
        results_dct['AUC'].append(ra_score)
    # plotting curve
    hue_range = HUE_VALS + np.array([0, HUE_STEP]) # we want to cover the entire range (including last val)
    hue_vals = np.arange(*hue_range, step=HUE_STEP)
    plot_curve(hue_vals, results_dct, hparams.output_path)

