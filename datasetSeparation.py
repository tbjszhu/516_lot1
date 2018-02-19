# the code is for generating test set and train set(base) #

import cv2
from utils import getImageListFromDir
import glob
import random
import os

### definition ###

data_dir = "/home/yxu/Downloads/png4_col/" # data dir

train_path = "./train_col/" # where to save train images
test_path = "./test_col/" # where to save test images

# create train directory
if not os.path.isdir(train_path):
    os.makedirs(train_path)
    print 'create ' + train_path


# create test directory
if not os.path.isdir(test_path):
    os.makedirs(test_path)
    print 'create ' + test_path


### Processing Separation of Images ###

def main():
    sub_dir = glob.glob(data_dir+'*/')  # all subdirs (classes) in the dataset
    sub_dir = sorted(sub_dir)  # sort file name
    print sub_dir

    for cls in sub_dir:

        imgs_addr = getImageListFromDir(cls, 'png')  # all images addr in a subset

        # separate images in a subset into train (2/3) and test (2/3)
        nb_train = 2*len(imgs_addr)/3
        nb_test = len(imgs_addr)-nb_train

        # save train images
        for i in range(nb_train):
            rd = random.randint(0, len(imgs_addr) - 1) # select randomly from the list
            img_path = imgs_addr[rd]
            filename = img_path.split("/")[-1]  # get filename
            print ('now putting img named '+filename +' to train set')
            img = cv2.imread(img_path)
            cv2.imwrite(train_path+filename, img)
            imgs_addr.pop(rd)

        # save test images
        for i in range(nb_test):
            rd = random.randint(0, len(imgs_addr) - 1) # select randomly from the list
            img_path = imgs_addr[rd]
            filename = img_path.split("/")[-1]  # get filename
            print ('now putting img named '+filename +' to test set')
            img = cv2.imread(img_path)
            cv2.imwrite(test_path+filename, img)
            imgs_addr.pop(rd)


main()



