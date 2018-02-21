## this code is for searching similar images for a given image ##

from utils import *
from sklearn.cluster import KMeans
from sklearn.externals import joblib
import cv2
import numpy as np
import os
import csv
import argparse

def main(train_addr, mode, descriptor_type, nfeatures, class_id):
    # definitions #
    model_dir = "./save_model/cv2_kmeans_mini_50_nf_100brief.pkl" # pretrained kmeans model for Brief 100 cluster
    target_addr = "./min_merged_test/251/rotation/251_c.png" # target image to search
    #target_addr = "./min_merged_test/252/luminence/252_i150.png"
    target_dir = "./min_merged_test/" # target dir to search
    hist_addr = './hists/'+descriptor_type # generated histograms for the dataset, if hist_addr = '', we will generate hists below
    # search similar images from base #
    kmeans = joblib.load(model_dir) # load pretrained kmeans model
    print ('kmeans parameters', kmeans.get_params())    

    # if hist_addr does not exist, generate hists for the dataset #
    if os.path.exists(hist_addr) == False:
        print "Creating Histogram..."
        os.makedirs(hist_addr)
        imgs_addr = getImageListFromDir(train_addr)
        for addr in imgs_addr:
            #print addr
            if descriptor_type == 'brief':
                img_gs = cv2.imread(addr, 0)
            else:
                img = cv2.imread(addr)
                img_gs = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            hist = generateHist(kmeans, img_gs, 'image', nfeatures, descriptor_type)
            filename = addr.split('/')[-1][0:-4]
            np.save(hist_addr + '/' + filename + '_' + descriptor_type, hist)
        has_hist = True
    else:
        print "Has Histogram"
        has_hist = True

    if (mode == 1):
        target = cv2.imread(target_addr)
        results, imgs_list = searchFromBase(hist_addr, target, kmeans, nfeatures, descriptor_type, class_id, has_hist)
        count = 1
        print results
        for key, value in results:
            filename = imgs_list[key].split('/')[-1]
            print ('NO. ' + str(count) +' is: ' + filename + ' distance : ' + str(value))
            count += 1
            
    elif (mode == 2): # try to find self rotated image from the database; rank the result; store in a csv file
        image_list = getImageListFromDir(target_dir)
        i = 0
        
        csv_file_path = './csv_result'
        csv_file_name = './csv_result/kmeans_' + str(kmeans.get_params()['n_clusters']) + '_nf_' + str(nfeatures) + descriptor_type + '.csv'
        if os.path.exists(csv_file_path) == False:
            os.mkdir(csv_file_path)
        csvfile = file(csv_file_name, 'wb')
        file_header = ['id','first_match','second_match','note']
        writer = csv.writer(csvfile)
        writer.writerow(file_header)

        note_global = 0 # global note for all the images in this classification
        first_count_global = 0
        second_count_global = 0
        count_control = 0
        for target, target_filename in img_generator(image_list):
            target_filename = target_filename.split('.')[0]
            target_fileno = target_filename.split('_')[0]
            #print target_fileno
            
            results, imgs_list = searchFromBase(hist_addr, target, kmeans, nfeatures, descriptor_type, has_hist=True)
            
            note = 10
            note_total = 0 # note for one image
            count = 1
            first_match = 0 # 0 means not found, otherwise means the ranking of the same object
            second_match = 0
            
            for key, value in results:
                filename = imgs_list[key].split('/')[-1]
                matched_fileno = filename.split('_')[0]
                if matched_fileno == target_fileno:
                    note_total =  note_total + note
                    if first_match == 0:
                        first_match = count # the ranking of the self first match
                        first_count_global += 1
                    else:
                        second_match = count
                        second_count_global += 1
                #print ('NO. ' + str(count) +' is: ' + matched_fileno + ' distance : ' + str(value))
                count += 1
                note -= 1
            writer.writerow([str(target_filename), str(first_match), str(second_match), str(note_total)])
            note_global += note_total
            
        writer.writerow(['Conclusion', str(first_count_global), str(second_count_global), str(note_global)])
        csvfile.close()
    elif (mode == 3): # generate for all class
        pr_csv_generation(target_dir, hist_addr, kmeans, nfeatures, descriptor_type)
    elif (mode == 4): # generate for only one class
        pr_csv_generation(target_dir, hist_addr, kmeans, nfeatures, descriptor_type, class_id)
    else:
        print "mode error should be [1~3]"
                
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", type=int, default=50,
                        help="Number of feature point for each image.")
    parser.add_argument("-c", type=int, default=25,
                        help="Number of cluster for kmeans")
    parser.add_argument("-d", type=str, default='orb',
                        help="Descriptor Type")
    parser.add_argument("-m", type=int, default=3,
                        help="Execution Mode")                                               
    parser.add_argument("--addr", type=str, default='./min_merged_train/',
                        help="training set addr")
    parser.add_argument("--tid", type=str, default=255,
                        help="test image class id for mode 4")

    args = parser.parse_args()
    
    train_addr = args.addr # './min_merged_train/' # path where train images lie
    nfeatures = args.n # 200 # Max quantity of kp, 0 as invalid for brief
    descriptor_type = args.d
    mode = args.m
    class_id = args.tid # 255 #target_addr.split('/')[-1].split('_')[0]    
    print "train_addr : %s, desptype : %s, nfeatures : %d" % (train_addr, descriptor_type, nfeatures)    
    main(train_addr, mode, descriptor_type, nfeatures, class_id)
    
