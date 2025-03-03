import os
import random
import numpy as np
import cv2

from torch.utils.data import Dataset
from utils import hwc_to_chw, read_img
from utils.Redegraded import adjust


def augment(imgs=[], size=[256, 256], edge_decay=0., only_h_flip=False):
    H, W, _ = imgs[0].shape
    Hc, Wc = size[0], size[1]

    if random.random() < Hc / H * edge_decay:
        Hs = 0 if random.randint(0, 1) == 0 else H - Hc
    else:
        Hs = random.randint(0, H-Hc)
    if random.random() < Wc / W * edge_decay:
        Ws = 0 if random.randint(0, 1) == 0 else W - Wc
    else:
        Ws = random.randint(0, W-Wc)

    for i in range(len(imgs)):
        imgs[i] = imgs[i][Hs:(Hs+Hc), Ws:(Ws+Wc), :]

    # horizontal flip
    if random.randint(0, 1) == 1:
        for i in range(len(imgs)):
            imgs[i] = np.flip(imgs[i], axis=1)

    if not only_h_flip:
        rot_deg = random.randint(0, 3)
        for i in range(len(imgs)):
            imgs[i] = np.rot90(imgs[i], rot_deg, (0, 1))

    return imgs


def align(imgs=[], size=256):
    H, W, _ = imgs[0].shape
    Hc, Wc = size[0], size[1]

    Hs = (H - Hc) // 2
    Ws = (W - Wc) // 2
    for i in range(len(imgs)):
        imgs[i] = imgs[i][Hs:(Hs+Hc), Ws:(Ws+Wc), :]

    return imgs


class PairLoader(Dataset):
    def __init__(self, data_dir, sub_dir, mode, size, edge_decay=0, only_h_flip=False):
        assert mode in ['train', 'valid', 'test']

        self.mode = mode
        self.size = size
        self.edge_decay = edge_decay
        self.only_h_flip = only_h_flip

        self.root_dir = os.path.join(data_dir, sub_dir)
        self.img_names = sorted(os.listdir(os.path.join(self.root_dir, 'GT')))  # 结尾加[:x]表示只取前x个
        self.img_num = len(self.img_names)

    def __len__(self):
        return self.img_num

    def __getitem__(self, idx):
        cv2.setNumThreads(0)
        cv2.ocl.setUseOpenCL(False)

        # read image, and scale [0, 1] to [-1, 1]
        img_name = self.img_names[idx]
        source_img = read_img(os.path.join(self.root_dir, 'IN', img_name)) * 2 - 1
        target_img = read_img(os.path.join(self.root_dir, 'GT', img_name)) * 2 - 1

        if self.mode == 'train':
            [source_img, target_img] = augment([source_img, target_img], self.size)

        if self.mode == 'valid':
            [source_img, target_img] = align([source_img, target_img], self.size)

        return {'input': hwc_to_chw(source_img), 'gt': hwc_to_chw(target_img), 'filename': img_name}


class SingleLoader(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.img_names = sorted(os.listdir(self.root_dir))
        self.img_num = len(self.img_names)

    def __len__(self):
        return self.img_num

    def __getitem__(self, idx):
        cv2.setNumThreads(0)
        cv2.ocl.setUseOpenCL(False)

        # read image, and scale [0, 1] to [-1, 1]
        img_name = self.img_names[idx]
        img = read_img(os.path.join(self.root_dir, img_name)) * 2 - 1

        return {'img': hwc_to_chw(img), 'filename': img_name}


class MultiLoader(Dataset):
    def __init__(self, data_dir, sub_dir, size):
        self.size = size
        self.root_dir = os.path.join(data_dir, sub_dir)
        self.img_names = sorted(os.listdir(os.path.join(self.root_dir, 'GT')))
        self.img_num = len(self.img_names)

    def __len__(self):
        return self.img_num

    def __getitem__(self, idx):
        cv2.setNumThreads(0)
        cv2.ocl.setUseOpenCL(False)
        img_name = self.img_names[idx]
        random.random()
        if random.randint(0, 1) == 0:
            input = read_img(os.path.join(self.root_dir, 'IN', img_name)) * 2 - 1
        else:
            input = adjust(os.path.join(self.root_dir, 'IN', img_name), over_enhance=False) * 2 - 1

        gt = read_img(os.path.join(self.root_dir, 'GT', img_name)) * 2 - 1
        cl1 = adjust(os.path.join(self.root_dir, 'GT', img_name), over_enhance=False, is_cl=True) * 2 - 1
        cl2 = adjust(os.path.join(self.root_dir, 'GT', img_name), over_enhance=True, is_cl=True) * 2 - 1
        [input, gt, cl1, cl2] = augment([input, gt, cl1, cl2], self.size)

        return {'input': hwc_to_chw(input), 'cl1': hwc_to_chw(cl1), 'cl2': hwc_to_chw(cl2),
                'gt': hwc_to_chw(gt), 'filename': img_name}

