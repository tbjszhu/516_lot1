### this python module aims for creating visualWords from clustering the base. ###
#!/usr/bin/python
# -*- coding: UTF-8 -*-
import numpy as np
import glob
from sklearn.cluster import KMeans
from sklearn.externals import joblib
from utils import generator_descriptor
import os
import argparse
import cv2


def main(train_addr, desptype, nfeatures, n_clusters):

    descpts_addr = "" # path where are saved the descriptors
    
    # read or generate local descriptors from the base (saved as numpy array). #
    # if descriptors not exist, create them here
    if descpts_addr == '':
        descpts_addr = "./dscpt_32bits_" + desptype
        if os.path.exists(descpts_addr) == False:
            os.mkdir(descpts_addr)
        generator_descriptor(train_addr, descpts_addr, nfeatures, desp_type=desptype)
        descpts_addr = descpts_addr + '/'+ 'nf_' + str(nfeatures) + "/*.npy" # path where are saved the descriptors
    else:
        descpts_addr += desptype
        descpts_addr = descpts_addr + '/' + 'nf_' + str(nfeatures) + "/*.npy"  # path where are saved the descriptors

    # each npy file has all local descriptors for an image
    descpts_list = glob.glob(descpts_addr)
    
    # generate train data for k means clustering of shape [num of descriptors, dimension of descriptor]
    train_data = None
    files_no_despt = []  # image list that have no key points that can be detected(too dark)
    for strr in descpts_list:
        if strr.find('kp') == -1: # not keypoints array
            if np.load(strr).shape == ():
                continue
            if train_data is None:  # first arr
                tmp = np.load(strr)
                train_data = tmp
                print "train_data shape", train_data.shape   
            elif np.load(strr).shape != () and np.load(strr).shape[1] == train_data.shape[1]:
                tmp = np.load(strr)
                train_data = np.vstack((train_data, np.load(strr)))

            else:
                files_no_despt.append(strr)  # images that we cannot extract keypoints and descriptors

    print "train size ", train_data.shape
    print files_no_despt

    # k-means clustering for train_data
    kmeans = KMeans(n_clusters, random_state=0).fit(train_data)
    if os.path.exists('./save_model') == False:
        os.mkdir('./save_model')
    if cv2.__version__[0] == '3':
        prefix = 'cv3'
    else:
        prefix = 'cv2'
        
    # save k-means model for further use
    joblib.dump(kmeans, './save_model/' + prefix + '_kmeans_mini_' + str(n_clusters) + '_nf_' + str(nfeatures) + desptype + '.pkl')


if __name__ == "__main__":
    # parameters entered by command lines
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", type=int, default=50,
                        help="Number of feature point for each image.")
    parser.add_argument("-c", type=int, default=25,
                        help="Number of cluster for kmeans")
    parser.add_argument("-d", type=str, default='orb',
                        help="Descriptor Type")                                               
    parser.add_argument("--addr", type=str, default='./min_merged_train/',
                        help="training set addr")                        

    args = parser.parse_args()
    
    train_addr = args.addr # './min_merged_train/' # path where train images lie
    desptype= args.d #'orb'  # type of descriptors to be generated
    nfeatures = args.n # Max quantity of kp
    n_clusters = args.c # Quantity of cluste
    print "train_addr : %s, desptype : %s, nfeatures : %d, nclusters : %d " % (train_addr, desptype, nfeatures, n_clusters)
    main(train_addr, desptype, nfeatures, n_clusters)

