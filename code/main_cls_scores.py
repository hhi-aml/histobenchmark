from tqdm import tqdm
import json
import numpy as np
import torch
import os
import sys
import torchmetrics.functional as F_metrics
FILE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(FILE_DIR)
import main_patho_lightning
from utils import bootstrap_utils


def collect_scores(model, dloader, cuda=True):
    if cuda:
        model = model.cuda()
    model = model.eval()
    with torch.no_grad():
        scores = torch.zeros([len(dloader.dataset), len(dloader.dataset.labels)])
        targets = torch.zeros(len(dloader.dataset), dtype=int)
        i = 0
        for x, y in tqdm(dloader, total=len(dloader)):
            if cuda:
                x = x.cuda()
            batch_scores = model(x)
            scores[i: i+len(batch_scores)] = batch_scores.cpu()
            targets[i: i+len(batch_scores)] = y.cpu()
            i += len(batch_scores)
    return scores.numpy(), targets.numpy()


def parser():
    """
    :return: The agrsparsing arguments, arguments of main_patho_lightning.py are kept, 
             only the xai-method argument was added 
    """
    ap = main_patho_lightning.add_default_args()
    ap = main_patho_lightning.add_model_specific_args(ap)
    return ap


if __name__ == '__main__':
    hparams = parser().parse_args()
    # checking if there is dset_paths.json in key2ai-med/meta_data/, if so dataset paths can just be read from there
    hparams = main_patho_lightning.dset_shortcut(hparams)
    # automaticly setting the input size for transformers
    hparams = main_patho_lightning.transformer_input_size_sanity(hparams)

    # loading the model
    model = main_patho_lightning.construct_model(hparams, requires_hooks=False)

    # loading data
    test_dloader = model.test_dataloader()
    #train_dloader = model.train_dataloader()

    # gather scores
    test_scores, test_targets = collect_scores(model, test_dloader, cuda=hparams.gpus>0)
    #train_scores, train_targets = collect_scores(model, train_dloader, cuda=hparams.gpus>0)

    # output sanity
    if not os.path.isdir(hparams.output_path):
        os.makedirs(hparams.output_path)
    # saving the scores
    #np.save(os.path.join(hparams.output_path, 'training_scores.npy'), train_scores)
    #np.save(os.path.join(hparams.output_path, 'training_targets.npy'), train_targets)
    np.save(os.path.join(hparams.output_path, 'test_scores.npy'), test_scores)
    np.save(os.path.join(hparams.output_path, 'test_targets.npy'), test_targets)

    # bootstrap results
    def F_acc(Y_hat, Y): return F_metrics.accuracy(torch.argmax(Y_hat, dim=1), Y)
    def F_auroc(Y_hat, Y): return F_metrics.auroc(torch.softmax(Y_hat, dim=1), Y, num_classes=hparams.num_classes, average='macro')
    input_tuple = tuple(map(torch.from_numpy, [test_scores, test_targets]))
    metrices_dict = {
        'accuracy': F_acc,
        'ROC': F_auroc
    }
    bootstrap_results = {}
    for name, score_fn in metrices_dict.items():
        print(f'{name}:\n{8 * "-"}\n')
        point, low, high, ids, results = bootstrap_utils.empirical_bootstrap(input_tuple, score_fn)
        print(f'point:  {point:.4f}\nlow:    {low:.4f}\nhigh:   {high:.4f}\n')
        if not name == 'accuracy':
            print(f'{point:.4f} \pm {max(point-low, high-point):.4f}')
        else:
            print(f'{100*point:.2f} \pm {100*max(point-low, high-point):.2f}')
        bootstrap_results[name] = (point, results.tolist())
    with open(os.path.join(hparams.output_path, 'bootstrap_results.json'), 'w') as f_out:
        json.dump(bootstrap_results, f_out)

