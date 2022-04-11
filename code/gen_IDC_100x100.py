# e.g. python3 idc_reconstruct.py /media/data/kaggle/IDC_regular_ps50_idx5/ /media/data/kaggle_big/ /media/data/IDC_100x100
import torch
from torchvision.datasets.folder import default_loader
from torchvision.utils import save_image
import torchvision.transforms.functional as F_trafo
import torchmetrics.functional as F_metric 
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from argparse import ArgumentParser


def tensor_loader(pth, H=None, W=None):
    if H is not None and W is not None:
        f_out = lambda tensor: F_trafo.resize(tensor, [H, W])
    else:
        f_out = lambda x: x
    return F_trafo.to_tensor(f_out(default_loader(pth)))


def extract_coords(pth):
    fltr = lambda p, var: int(p.split(var)[1].split('_')[0])
    return torch.tensor([fltr(pth, '_y'), fltr(pth, '_x')], dtype=int)


def reconstruct(foldrs, wsis, suffixes=['.png', '.jpg', '.jpeg']):
    foldrs_patches = []
    foldrs_labels = []
    foldrs_rows = []
    foldrs_cols = []
    p_bar = tqdm(enumerate(foldrs), total=len(foldrs))
    for d_idx, d in p_bar:
        p_bar.set_description(d)
        new_patches = []
        new_labels = []
        patch_rows = []
        patch_cols = []
        paths = []
        for r, _, fs in os.walk(d):
            for f in fs:
                if any([f.endswith(s) for s in suffixes]):
                    paths.append(os.path.join(r, f))
        N = len(paths)
        C, H, W = tensor_loader(paths[0]).shape
        # gather patches / labels / coordinates
        patches = torch.zeros([N, C, H, W])
        labels = torch.zeros([N])
        coordinates = torch.zeros([N, 2], dtype=int)
        for i, pth in enumerate(paths):
            tensor = tensor_loader(pth, H, W)
            lbl = int(os.path.basename(os.path.dirname(pth)))
            coords = extract_coords(pth)
            labels[i] = lbl
            patches[i] = tensor
            coordinates[i] = coords
        # TODO: fancy schmanzy reconstruction magic -> utilize original WSI etc.
        rows = int((coordinates[:,0].max() + H).item())
        columns = int((coordinates[:,1].max() + W).item())
        # step 1 reconsturct image
        img = torch.zeros([C, rows, columns]) 
        original_img = tensor_loader(wsis[d_idx])
        img_lbls = torch.zeros([rows, columns])
        for l, p, (r, c) in zip(labels, patches, coordinates):
            img[:, r:r+H, c:c+W] = p
            img_lbls[r:r+H, c:c+W] = l
        del patches
        del labels
        del coordinates
        # step 2 search for fitting windows
        marked = torch.zeros([rows, columns], dtype=bool)
        overlay = torch.zeros([rows, columns]) 
        for r in range(0, rows-2*H, 2*H):
            for c in range(0, columns-2*W, 2*W):
                if (img[:, r:r+2*H, c:c+2*W] > 0).sum() >= 0.5 * H * W:
                    new_patches.append(original_img[:, r:r+2*H, c:c+2*W])
                    new_labels.append(int(img_lbls[r:r+2*H, c:c+2*W].sum() > 0))
                    patch_rows.append(r)
                    patch_cols.append(c)
        #             overlay[r:r+2*H, c:c+2*W] = 1
        # img[img==0] = 1
        # plt.imshow(original_img.numpy().transpose(1,2,0))
        # plt.imshow(overlay.numpy(), alpha=0.5, cmap='Greens')
        # plt.show()
        # plt.imshow(img.numpy().transpose(1,2,0))
        # plt.imshow(overlay.numpy(), alpha=0.5, cmap='Greens')
        # plt.show()
        foldrs_patches.append(new_patches)
        foldrs_labels.append(new_labels)
        foldrs_rows.append(patch_rows)
        foldrs_cols.append(patch_cols)
    return foldrs_patches, foldrs_labels, foldrs_rows, foldrs_cols


def save_imgs(patches, labels, folders, rows, cols, output_dir):
    for ps, lbls, rs, cs, d in zip(patches, labels, rows, cols, folders):
        d = os.path.basename(d)
        for subdir in '01':
            if not os.path.isdir(os.path.join(output_dir, d, subdir)):
                os.makedirs(os.path.join(output_dir, d, subdir))
        for p, lbl, r, c in zip(ps, lbls, rs, cs):
            save_image(p, os.path.join(output_dir, d, str(lbl), f'{d}_class{lbl}_row{r}_col{c}.png'))


def parser():
    ap = ArgumentParser()
    ap.add_argument('j_data')
    ap.add_argument('orig')
    ap.add_argument('output_dir')
    return ap


if __name__ == '__main__':
    args = parser().parse_args()
    j_data = args.j_data
    orig = args.orig
    output_dir = args.output_dir
    foldrs = [os.path.join(j_data, d) for d in os.listdir(j_data) if os.path.isdir(os.path.join(j_data, d))]
    all_wsis = []
    for r, _, fs in os.walk(orig):
        for f in fs:
            if f.endswith('png'):
                all_wsis.append(os.path.join(r, f))
    # matching wsi to j-data
    wsis = []
    for f in foldrs:
        wsi_id = os.path.basename(f)
        for w in all_wsis:
            if os.path.basename(w).split('_')[0] == wsi_id and 'mask' not in w:
                wsis.append(w)
                break
    # fetching new patches
    patches, labels, rows, cols = reconstruct(foldrs, wsis)
    save_imgs(patches, labels, foldrs, rows, cols, output_dir)

