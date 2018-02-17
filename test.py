## this code is for searching similar images for a given image ##

from utils import searchFromBase, generateHist, getImageListFromDir
from sklearn.cluster import KMeans
from sklearn.externals import joblib
import cv2
import numpy as np
import os

def main():
    # definitions #

    base_dir = "./train/" # base dir for search
    model_dir = "./save_model/kmeans.pkl" # pretrained kmeans model
    target_dir = "./test/514_c.png" # target to search
    hist_addr = ''  # generated histograms for the dataset, if hist_addr = '', we will generate hists below
    descriptor_tpye = 'orb'

    # search similar images from base #
    kmeans = joblib.load(model_dir) # load pretrained kmeans model
    print ('kmeans parameters', kmeans.get_params())

    # if hist_addr does not exist, generate hists for the dataset #
    if hist_addr == '':
        hist_addr = './hists/'
        if os.path.exists(hist_addr) == False:
            os.mkdir(hist_addr[0:-1])
        imgs_addr = getImageListFromDir(base_dir)
        for addr in imgs_addr :
            print addr
            img = cv2.imread(addr)
            img_gs = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            hist = generateHist(kmeans, img_gs, 'image', descriptor_tpye)
            filename = addr.split('/')[-1][0:-4]
            print filename
            np.save(hist_addr+filename+'_'+descriptor_tpye, hist)

    target = cv2.imread(target_dir)
    results, imgs_list = searchFromBase(hist_addr, target,kmeans, has_hist=True)
    count = 1
    for key, value in results:
        filename = imgs_list[key].split('/')[-1]
        print ('NO. ' + str(count) +' is: ' + filename)
        print ("with distance from the target : " + str(value))
        count += 1

if __name__ == '__main__':
    main()
