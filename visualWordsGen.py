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
    train_addr = './train/' # path where train images lie
    descpts_addr = ''  # path where are saved the descriptors, If descripts_addr = '', create them below
    desptype='orb'  # type of descriptors to be generated


    # read or generate local descriptors from the base (saved as numpy array). #

    # if descriptors not exist, create them here !
    if descpts_addr == '':
        descpts_addr = "./dscpt_32bits_orb"
        if os.path.exists(descpts_addr) == False:
            os.mkdir(descpts_addr)
        generator_descriptor(train_addr, descpts_addr, desp_type=desptype)
        descpts_addr = descpts_addr+"/*.npy" # path where are saved the descriptors

    # each npy file has all local descriptors for an image
    descpts_list = glob.glob(descpts_addr)
    print descpts_list

    # generate train data for k means clustering of shape [num of descriptors, dimension of descriptor]
    train_data = None
    files_no_despt = []
    for strr in descpts_list:
        ret = strr.find('kp')
        if strr.find('kp') == -1: # not keypoints array

            if train_data is None:  # first arr
                tmp = np.load(strr)
                train_data = tmp

            elif np.load(strr).shape != () and np.load(strr).shape[1] == train_data.shape[1]:
                print strr
                tmp = np.load(strr)
                train_data = np.vstack((train_data, np.load(strr)))

            else:
                files_no_despt.append(strr)  # images that we cannot extract keypoints and descriptors

    print train_data.shape
    print files_no_despt

    # k-means clustering for train_data
    kmeans = KMeans(n_clusters=400, random_state=0).fit(train_data)
    print kmeans.labels_
    print kmeans.cluster_centers_
    if os.path.exists('./save_model') == False:
        os.mkdir('./save_model')
    joblib.dump(kmeans, './save_model/kmeans_400.pkl')
    #joblib.dump(kmeans, './save_model/kmeans_'+str(kmeans.get_params()['n_clusters'])+'.pkl')

main()


