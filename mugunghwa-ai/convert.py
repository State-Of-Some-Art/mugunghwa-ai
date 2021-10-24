from PIL import Image
import cv2
import numpy as np
import torch

def CHW2HWC(src):
    src = np.array(src)
    src = src.transpose(1, 2, 0)
    return src

def HWC2CHW(src):
    src = np.array(src)
    src = src.transpose(2, 0, 1)
    return src

def normalize(src, min=0, max=255):
    src = np.array(src)
    src = src - min
    src = src / (max - min)
    return src

def to_8bit(src, min=0, max=255):
    src = normalize(src, min, max)
    src = (src * 255).astype(np.uint8)
    return src