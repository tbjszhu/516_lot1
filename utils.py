import numpy as np
import cv2
from matplotlib import pyplot as plt
import glob

# Read images/masks from a directory

def getImageListFromDir(img_dir, filetype='png'):
    """
    :param img_dir: imgs root dir, string
    :param filetype: "png", "jpg" or "bmp", string
    :return: list of images, list
    """
    img_dir = img_dir + '*.'+filetype
    print img_dir
    l = sorted(glob.glob(img_dir))
    return l

    return None

# read images with img generator

def img_generator(img_list):
    """
    :param img_list: list of iamges, list of string
    :return: yield a pair of sequences images
    """
    while len(img_list)> 0:
        f1 = img_list.pop(0)
        print("read img: ", f1.split('/')[-1])
        img = cv2.imread(f1)
        yield img