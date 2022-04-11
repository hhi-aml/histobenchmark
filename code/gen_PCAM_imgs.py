"""
Sadly the PCAM dataset is provided as HDF5 files.
These files are not compatible with the PyTorch Dataloader Framework.
Therefore all data has to be extracted from the HDF5 dataset files and stored as conventional images and annotations.

To do so call this script and pass it the base directiory of all (unziped) PCAM HDF5 files.
e.g.:

python3 gen_PCAM.py /media/data/PCAM
"""
import pandas as pd
import os
import h5py
import cv2
import tqdm
from argparse import ArgumentParser


def parser():
    ap = ArgumentParser()
    ap.add_argument('base_dir')
    return ap


def gen_PCAM(base_dir):
    files_y = [
        'camelyonpatch_level_2_split_train_y.h5',
        'camelyonpatch_level_2_split_test_y.h5',
        'camelyonpatch_level_2_split_valid_y.h5'
    ]
    files_x = [
        'camelyonpatch_level_2_split_train_x.h5',
        'camelyonpatch_level_2_split_test_x.h5',
        'camelyonpatch_level_2_split_valid_x.h5'
    ]
    splits = ['train', 'test', 'valid']
    files = files_x + files_y
    dir_files = os.listdir(base_dir)
    # sanity
    if not all([f in dir_files for f in files]):
        raise LookupError(f'{base_dir} does not feature all required files: {files}')
    for s in splits:
        if not os.path.isdir(os.path.join(base_dir, s)):
            os.makedirs(os.path.join(base_dir, s))
    # parsing and writing files
    annotations = {'path': [], 'label': []}
    for f_x, f_y, split in zip(files_x, files_y, splits):
        X = h5py.File(os.path.join(base_dir, f_x), 'r')['x']
        Y = h5py.File(os.path.join(base_dir, f_y), 'r')['y']
        N = len(X)
        print(f'\nGenerating the {split} subset at: {os.path.join(base_dir, split)}')
        p_bar = tqdm.tqdm(total=N)
        for i, (x, y) in enumerate(zip(X, Y)):
            leading_0 = '0' * (len(str(N)) - len(str(i)))
            file_path = os.path.join(split, leading_0 + f'{i}.png')
            cv2.imwrite(os.path.join(base_dir, file_path), x)
            annotations['path'].append(file_path)
            annotations['label'].append(y.flatten()[0])
            p_bar.update(1)
    annotations_path = os.path.join(base_dir, 'annotations.csv')
    print(f'\nWriting annotations to: {annotations_path}')
    df = pd.DataFrame.from_dict(annotations)
    df.to_csv(annotations_path)


if __name__ == '__main__':
    ap = parser().parse_args()
    gen_PCAM(ap.base_dir)
