import numpy as np
import cv2
from matplotlib import pyplot as plt
import glob
import os

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

def generator_descriptor(fileaddr, save_addr, desp_type='orb'):
    """
    :param fileaddr: string, images dir
    :param save_addr: string, where to save
    :param desp_type: 'orb'
    :return: None
    """

    # create save directory
    if not os.path.isdir(save_addr):
        os.makedirs(save_addr)
        print 'create ' + save_addr

    fileList = os.listdir(fileaddr)
    for filename in fileList:
        if '.png' in filename:
            #print filename
            filename_des = filename.split('.')[0]
            img = cv2.imread(fileaddr+filename, 0)
            # Initiate STAR detector

            orb = []
            kp = []

            if desp_type == "orb" :
                if int(cv2.__version__[0]) == 3:
                    orb = cv2.ORB_create()
                elif int(cv2.__version__[0]) == 2:
                    orb = cv2.ORB()
                else:
                    print "opencv version error"
                    sys.exit(0)
                # find the keypoints with ORB
                kp = orb.detect(img, None)
                # compute the descriptors with ORB
                kp, des = orb.compute(img, kp)
                

            # creat file to store descriptor
            np.save(save_addr+'/'+filename_des, des)
            
            # save img with keypoint
            # img2 = cv2.drawKeypoints(img,kp,color=(0,255,0), flags=0)
            # v2.imwrite(save_addr+'/'+filename_des+'_addkp', img2)
