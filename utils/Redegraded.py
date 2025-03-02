import copy
import os
import cv2
import numpy as np
import random
from skimage import exposure
import os


def adjust(img, over_enhance=False, is_cl=False):
    input = cv2.imread(img)
    sub = 0 if is_cl is True else 0
    up = 0.25
    w, h, c = input.shape
    input_rgb = cv2.resize(input, dsize=None, fx=0.3, fy=0.3)  # 压缩输入
    Luminance = np.ones_like(input_rgb).astype(np.float)

    for sigma in [15, 60, 90]:
        Luminance1 = cv2.GaussianBlur(input_rgb, (0, 0), sigma).astype('float32')
        Luminance1[Luminance1 < 255] += 1e-2
        Luminance1 = np.log10(Luminance1)
        Luminance1 = np.clip(Luminance1, 0, 255)
        Luminance += Luminance1
    Luminance = Luminance / 3  # 保留原始光场以便修改, range in [0-4]

    # 先得到初始反射分量
    L = cv2.resize(Luminance, dsize=(h, w))
    L = (L - np.min(L)) / (np.max(L) - np.min(L) + 0.0001)
    L = np.uint8(L * 255)  # 入射分量
    R = cv2.subtract(np.int16(input), np.int16(L))  # 反射分量

    L1_blur = copy.deepcopy(Luminance)

    # 调整1——高斯模糊仿烟雾
    L1_blur = cv2.GaussianBlur(L1_blur, (0, 0), 3).astype('float32')
    L1_blur[L1_blur < 255] += 1e-2
    L1_blur = np.log10(L1_blur)  # 中间代表核大小，尾代表周围像素对中心的影响
    L1_blur = np.clip(L1_blur, 0, 255)
    L1_blur = cv2.resize(L1_blur, dsize=(h, w))
    L1_blur = (L1_blur - np.min(L1_blur)) / (np.max(L1_blur) - np.min(L1_blur) + 0.0001)
    L1_blur = np.int16(L1_blur * 255)

    # 调整2——色偏
    if over_enhance is not True:
        L1_blur[:, :, 2] = L1_blur[:, :, 2] * (1 - random.uniform(sub, up))
        L1_blur[:, :, 1] = L1_blur[:, :, 1] + (random.uniform(sub, up) * (255 - L1_blur[:, :, 1])/255)
        L1_blur[:, :, 0] = L1_blur[:, :, 0] + (random.uniform(sub, up) * (255 - L1_blur[:, :, 0])/255)
    else:
        L1_blur[:, :, 2] = L1_blur[:, :, 2] + (random.uniform(sub, up) * (255 - L1_blur[:, :, 2]) / 255)
        L1_blur[:, :, 1] = L1_blur[:, :, 1] * (1 - random.uniform(sub, up))
        L1_blur[:, :, 0] = L1_blur[:, :, 0] * (1 - random.uniform(sub, up))
    out1 = cv2.add(R, L1_blur)
    out1 = np.clip(out1, 0, 255)

    # 调整3——对比度
    out1 = cv2.convertScaleAbs(out1)
    i_hsv = cv2.cvtColor(out1, cv2.COLOR_BGR2HSV)

    rate = random.uniform(sub, up)
    if over_enhance is not True:
        i_hsv[:, :, 2] = i_hsv[:, :, 2] * (1 - rate)
    else:
        m = np.max(i_hsv[:, :, 2])
        i_hsv[:, :, 2] = i_hsv[:, :, 2] * (1 + rate * (m - i_hsv[:, :, 2]) / m)
    out2 = cv2.cvtColor(i_hsv, cv2.COLOR_HSV2BGR)
    out2 = np.clip(out2, 0, 255)
    return out2[:, :, ::-1].astype('float32') / 255.0
