### this python module aims for creating visualWords from clustering the base. ###
#!/usr/bin/python
# -*- coding: UTF-8 -*-
import numpy as np
import glob
from sklearn.cluster import KMeans
from sklearn.externals import joblib
from utils import generator_descriptor
import os
def main():
    # definitions #
    train_addr = './min_merged_train/' # path where train images lie
    descpts_addr = "" # path where are saved the descriptors, If descripts_addr = '', create them below
    desptype='orb'  # type of descriptors to be generated
    nfeatures = 50 # Max quantity of kp, 0 as invalid for brief
    pick_nfeatures = 50 # choose top pick_nfeatures

    # read or generate local descriptors from the base (saved as numpy array). #

    # if descriptors not exist, create them here !

    if descpts_addr == '':
        descpts_addr = "./dscpt_128bits_" + desptype
        if os.path.exists(descpts_addr) == False:
            os.mkdir(descpts_addr)
        generator_descriptor(train_addr, descpts_addr, nfeatures, desp_type=desptype)
        descpts_addr = descpts_addr + '/'+ 'nf_' + str(nfeatures) + "/*.npy" # path where are saved the descriptors
    else:
        descpts_addr += desptype
        descpts_addr = descpts_addr + '/' + 'nf_' + str(nfeatures) + "/*.npy"  # path where are saved the descriptors

    # each npy file has all local descriptors for an image
    descpts_list = glob.glob(descpts_addr)
    print descpts_list

    # generate train data for k means clustering of shape [num of descriptors, dimension of descriptor]
    train_data = None
    files_no_despt = []
    for strr in descpts_list:
        if strr.find('kp') == -1: # not keypoints array

            if train_data is None:  # first arr
                tmp = np.load(strr)[0:pick_nfeatures, :]
                train_data = tmp

            elif np.load(strr).shape != () and np.load(strr).shape[1] == train_data.shape[1]:
                print strr
                tmp = np.load(strr)[0:pick_nfeatures, :]
                print tmp.shape
                train_data = np.vstack((train_data, np.load(strr)))

            else:
                files_no_despt.append(strr)  # images that we cannot extract keypoints and descriptors

    print "train size ", train_data.shape
    print files_no_despt

    # k-means clustering for train_data
    n_clusters = 50
    kmeans = KMeans(n_clusters, random_state=0).fit(train_data)
    if os.path.exists('./save_model') == False:
        os.mkdir('./save_model')
    joblib.dump(kmeans, './save_model/opencv3_kmeans_mini_' + str(n_clusters) + '_nf_' + str(nfeatures) + desptype + '.pkl')
    #joblib.dump(kmeans, './save_model/kmeans_'+str(kmeans.get_params()['n_clusters'])+'.pkl')

main()


