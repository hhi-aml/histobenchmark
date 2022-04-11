from argparse import ArgumentParser
import numpy as np
import os
import json


def colored(s):
    return '\033[92m' + s + '\033[0m'


def parser():
    ap = ArgumentParser()
    ap.add_argument('pth_model_logs_A')
    ap.add_argument('--pth_model_logs_B', type=str, default=None)
    ap.add_argument('--thresh_pct', type=float, default=0.6)
    return ap


def load_experiments_bootstraps(pth):
    results = []
    keys = []
    for sub_dir in os.listdir(pth):
        with open(os.path.join(pth, sub_dir, 'bootstrap_results.json'), 'r') as f_json:
            stats = json.load(f_json)
            results.append(stats)
            keys += list(stats.keys())
    keys = set(keys)
    metrices = {k: [] for k in keys}
    for result in results:
        for k in result.keys():
            metrices[k].append(result[k])
    for k in keys:
        metrices[k] = ( np.array([m[0] for m in metrices[k]]), np.array([m[1] for m in metrices[k]]) )
        #print(metrices[k][0])
    return metrices


def report_stats(metrices):
    for m in sorted(metrices.keys()):
        print(colored(m))
        mean = np.mean(metrices[m][0])
        std = np.std(metrices[m][0])
        if m == 'accuracy':
            str_acc = rf'{mean*100:.2f} $\pm$ {std*100:.2f}'
            print(str_acc)
        else:
            str_ROC = rf'{mean:.4f} $\pm$ {std:.4f}'
            print(str_ROC)
    print(str_acc + ' & ' + str_ROC)


def compare(metrices_A, metrices_B, thresh_pct=0.8):
    report = {}
    result = {}
    for m in sorted(metrices_A.keys()):
        scores_A = metrices_A[m][1]
        scores_B = metrices_B[m][1]
        N = scores_A.shape[0]
        diffs = np.repeat(scores_A, N, axis=0) - np.vstack([scores_B for _ in range(N)])
        count = 0
        for diff in diffs:
            count += not_sign_worse(diff)
        report[m] = count / diffs.shape[0]
        result[m] = (count / diffs.shape[0]) >= thresh_pct
    return result, report


def not_sign_worse(diffs, alpha=0.95):
    low = np.percentile(diffs, ((1.0-alpha)/2.0) * 100)
    high = np.percentile(diffs, (alpha+((1.0-alpha)/2.0)) * 100)
    return int(low<=0)# and high>0)


if __name__ == '__main__':
    hparams = parser().parse_args()
    metrices_A = load_experiments_bootstraps(hparams.pth_model_logs_A)
    report_stats(metrices_A)
    if hparams.pth_model_logs_B is not None:
        metrices_B = load_experiments_bootstraps(hparams.pth_model_logs_B)
        result, report = compare(metrices_A, metrices_B, hparams.thresh_pct)
        for k in sorted(result.keys()):
            print(colored(f'{k} not significantly worse: {result[k]} ({report[k]})'))

