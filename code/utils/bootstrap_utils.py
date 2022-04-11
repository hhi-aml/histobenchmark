__all__ = ['empirical_bootstrap']

import numpy as np
from sklearn.utils import resample
from multiprocessing import Pool
from functools import partial
from tqdm import tqdm
import torch


def _eval(ids, input_tuple, score_fn):
    return score_fn(*[t[ids] for t in input_tuple])


def empirical_bootstrap(input_tuple, score_fn, ids=None, n_iterations=100, alpha=0.95, threads=None):
    if(not(isinstance(input_tuple,tuple))):
        input_tuple = (input_tuple,)
    score_point = score_fn(*input_tuple)
    if not isinstance(score_point, np.ndarray):
        score_point = float(score_point.numpy())

    if(n_iterations==0):
        return score_point, np.zeros(score_point.shape), np.zeros(score_point.shape), []

    if(ids is None):
        local_state = np.random.get_state()
        np.random.seed(42)
        ids = []
        for _ in np.arange(n_iterations):
            ids.append(resample(range(len(input_tuple[0])), n_samples=len(input_tuple[0])))
        ids = np.array(ids)
        np.random.set_state(local_state)


    # pool = Pool(1)#threads)
    fn = partial(_eval, input_tuple=input_tuple, score_fn=score_fn)
    # results = pool.map(fn, ids)
    # pool.close()
    # pool.join()
    results = []
    for sample_ids in ids:
        results.append(fn(sample_ids))
    results = np.array(results)
    score_diff = results - score_point

    score_low = score_point + np.percentile(score_diff, ((1.0-alpha)/2.0) * 100)
    score_high = score_point + np.percentile(score_diff, (alpha+((1.0-alpha)/2.0)) * 100)

    return score_point, score_low, score_high, ids, results


def bootstrap_model_eval(model, dataloader, score_fn, **kwargs):
    Y = []
    Y_hat = []
    with torch.no_grad():
        model.eval()
        for x, y in tqdm(dataloader):
            Y_hat.append(model(x.to(model.device)).detach().cpu())
            Y.append(y.detach().cpu().long())
    input_tuple = tuple(map(torch.cat, (Y_hat, Y)))
    return empirical_bootstrap(input_tuple, score_fn, **kwargs)


def bootstrap_models_compare(model_1, model_2, dataloader, score_fn, **kwargs):
    Y_hat_1 = []
    Y_hat_2 = []
    with torch.no_grad():
        model.eval()
        for x, y in tqdm(dataloader):
            Y_hat_1.append(model_1(x.to(model_1.device)).detach().cpu())
            Y_hat_2.append(model_2(x.to(model_2.device)).detach().cpu())
    input_tuple = tuple(map(torch.cat, (Y_hat_1, Y_hat_2)))
    return empirical_bootstrap(input_tuple, score_fn, **kwargs)

