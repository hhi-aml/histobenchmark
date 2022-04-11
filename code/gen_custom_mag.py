from argparse import ArgumentParser
import os
import numpy as np
import cv2
import sys
from tqdm import tqdm
FILE_DIR = os.path.dirname(os.path.abspath(__file__))


def calc_patch_shape(h_orig, w_orig, mag_orig, mag_new):
    scale_factor = 1 / np.sqrt(mag_new / mag_orig)
    return int(h_orig * scale_factor), int(w_orig * scale_factor)


def extract_patches(img, mag_orig, mag_new, convention='shift'):
    H = img.shape[0]
    W = img.shape[1]
    patch_shape = calc_patch_shape(H, W, mag_orig, mag_new)
    offset_h = 0
    offset_w = 0
    patches = []
    w_done = False
    h_done = False
    while not (w_done and h_done):
        # resetting column position
        w_done = False
        offset_w = 0
        # column-wise extraction of all patches from a row
        while not w_done:
            # dealing with imperfect fits when sliding the patch window
            if offset_h+patch_shape[0] > H or offset_w+patch_shape[1] > W:
                # moving to next row after width overflow
                if offset_w+patch_shape[1] > W:
                    w_done = True
                # applying the convention
                if convention == 'drop':
                    w_done = True
                    break
                elif convention == 'shift':
                    if offset_h + patch_shape[0] > H:
                        h_start, h_max = H - patch_shape[0], H
                    else:
                        h_start, h_max = offset_h, offset_h + patch_shape[0]
                    if offset_w + patch_shape[1] > W:
                        w_start, w_max = W - patch_shape[1], W
                    else:
                        w_start, w_max = offset_w, offset_w + patch_shape[1]
                    patches.append(img[h_start:h_max, w_start:w_max])
                elif convention == 'pad':
                    padded = np.zeros(patch_shape.shape)
                    extracted = img[offset_h : min(H, offset_h+patch_shape[0]), offset_w : min(W, img.shape[1])]
                    padded[:extracted.shape[0], :extracted.shape[1]] = extracted
                    patches.append(padded)
                elif convention == 'stretch':
                    patches.append(np.resize(
                        img[offset_h : min(H, offset_h+patch_shape[0]), offset_w : min(W, offset_w+patch_shape[1])], 
                        patch_shape
                    ))
                else:
                    raise NotImplementedError(f'unknown overflow convention {convention}')
            # patch fits neatly
            else:
                patches.append(img[offset_h : offset_h+patch_shape[0], offset_w : offset_w+patch_shape[1]])
            # moving to the new column
            offset_w += patch_shape[1]
        if offset_h > H:
            h_done = True
            break
        # moving to the next row
        offset_h += patch_shape[0]
    return patches


def search(root, suffixes=['.png', '.tif', '.jpg', '.bmp']):
    if not os.path.isdir(root):
        raise ValueError(f'{root} is not a directory')
    if not root.endswith('/'):
        root += '/'
    files = []
    for r, _, fs in os.walk(root):
        for f in fs:
            if any([f.endswith(sfx) for sfx in suffixes]):
                files.append(os.path.join(r, f))
    return files, [f.split(root)[1] for f in files]


def f_name_extend(f_name, extension, suffix='.png'):
    return f_name.split('.')[0] + extension + suffix


def arg_parser():
    ap = ArgumentParser()
    ap.add_argument('img_dir', help='directory with images to extract patches from', type=str)
    ap.add_argument('out_dir', help='directory to store patches in', type=str)
    ap.add_argument('--old-mag', default=40, type=int, help='magnification of images')
    ap.add_argument('--new-mag', default=400, type=int, help='new magnification of patches')
    ap.add_argument('--convention', default='drop', type=str,
                    help='convention when dealing with overflow {drop, shift, pard, stretch}')
    return ap


if __name__ == '__main__':
    args = arg_parser().parse_args()
    # sanity
    if not os.path.isdir(args.out_dir):
        os.makedirs(args.out_dir)
    assert args.old_mag < args.new_mag, "New magnification has to be larger than that of the images"
    # searching for images
    img_paths, rel_paths = search(args.img_dir)
    # generating patches
    img_patches = []
    img_patch_names = []
    print('extracting patches')
    for img_p, rel_p in tqdm(list(zip(img_paths, rel_paths))):
        img = cv2.imread(img_p)
        extracted = extract_patches(img, args.old_mag, args.new_mag, convention=args.convention)
        img_patches += extracted
        img_patch_names += [os.path.join(args.out_dir, f_name_extend(rel_p, f'_patch_{i}'))
                            for i in range(len(extracted))]
    # writing patches
    print(f'writing patch images to {args.out_dir}')
    for patch, f_name in tqdm(list(zip(img_patches, img_patch_names))):
        cv2.imwrite(f_name, patch)

