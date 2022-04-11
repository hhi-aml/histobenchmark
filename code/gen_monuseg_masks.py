from tqdm import tqdm
import os
import numpy as np
from skimage.draw import polygon
import xml.etree.ElementTree as ET
from torchvision.datasets.folder import default_loader
import cv2
from argparse import ArgumentParser


def gen_masks(d_xml, d_imgs, d_out, suffix_imgs='.tif'):
    # filter for xml and image files
    xmls = [f for f in os.listdir(d_xml) if f.endswith('.xml')]
    imgs = [f for f in os.listdir(d_imgs) if f.endswith(suffix_imgs)]
    # put list in same order
    xmls = sorted(xmls, key=lambda f: f[:-4])
    imgs = sorted(imgs, key=lambda f: f[:-len(suffix_imgs)])
    # sanity check
    if not os.path.isdir(d_out):
        os.makedirs(d_out)
    # write masks and images to d_out
    for f_img, f_xml in tqdm(zip(imgs, xmls), total=len(imgs)):
        img = np.array(default_loader(os.path.join(d_imgs, f_img)))
        mask = xml_to_mask(os.path.join(d_xml, f_xml), img.shape)
        cv2.imwrite(os.path.join(d_out, f_img), img)
        cv2.imwrite(os.path.join(d_out, f_xml[:-4]+'_mask.tif'), mask)


def xml_to_mask(file_path, img_shape):
    # gather list of outlines/ vertices
    tree = ET.parse(file_path)
    root = tree.getroot()
    regions = root[0][1] # root -> Annotation -> [..., Regions]
    region_vertices = []
    for r in regions: # Regions -> [Region, Region, ...]
        region_vertices.append(r[1]) # Region -> [..., Vertices]
    # convert to coordinates
    coords = []
    for vs in region_vertices:
        tmp = []
        for v in vs:
            tmp.append([v.get('Y'), v.get('X')])
        if tmp != []:
            coords.append(np.array(tmp, dtype=float))
    # filll out polygons
    idcs = []
    for p in coords:
        idcs.append(polygon(p[:,0], p[:,1], img_shape))
    img = np.zeros(img_shape, dtype=np.uint8)
    for rr, cc in idcs:
        img[rr, cc, :] = 255
    return img


def parser():
    ap = ArgumentParser()
    ap.add_argument('dir_xml', type=str, help='directory where .xml annotations are stored.')
    ap.add_argument('dir_imgs', type=str, help='directory where .tif images are stored')
    ap.add_argument('out', type=str, help='directory to write output to.')
    return ap

if __name__ == '__main__':
    args = parser().parse_args()
    gen_masks(args.dir_xml, args.dir_imgs, args.out)

