## this code is for searching similar images for a given image ##

from utils import *
from sklearn.cluster import KMeans
from sklearn.externals import joblib
import cv2
import numpy as np
import os
import csv
import argparse
import shutil


def main(train_addr, mode, descriptor_type, nfeatures, class_id, target_addr):
    # definitions #
    model_dir = "./save_model/cv2_kmeans_mini_50_nf_100orb.pkl" # pretrained kmeans model for Brief 100 cluster
    target_dir = "./min_merged_test/" # target dir to search
    hist_addr = './hists/'+descriptor_type # generated histograms for the dataset
    

    # load pre-trained k-means model #
    kmeans = joblib.load(model_dir) 
    print ('kmeans parameters', kmeans.get_params())    

    # if hist_addr does not exist, generate hists for the data set
    if os.path.exists(hist_addr) == False:
        print "Creating Histograms..."
        os.makedirs(hist_addr)
        imgs_addr = getImageListFromDir(train_addr)
        for addr in imgs_addr:
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
        print "Histograms exist"
        has_hist = True

    # test for one image #
    if (mode == 1):
        if descriptor_type == 'brief':
            target = cv2.imread(target_addr, 0) 
        else:   
            target = cv2.imread(target_addr)
        # search similar images from base and get the ranking results  
        results, imgs_list = searchFromBase(hist_addr, target, kmeans, nfeatures, descriptor_type, mode, class_id, has_hist)
        
        # draw the top ranking results in a 3x4 figure
        ax = [0] * 12
        f,((ax[0],ax[1], ax[2], ax[3]),(ax[4],ax[5], ax[6], ax[7]), (ax[8],ax[9], ax[10], ax[11])) = plt.subplots(3,4)
        original = cv2.imread(target_addr) 
        ax[0].set_title("Org: " + target_addr.split('/')[-1].split('.')[0])
        original = cv2.cvtColor(original,cv2.COLOR_BGR2RGB)
        ax[0].imshow(original)
        ax[0].set_axis_off()
        count = 1                
        for key, value in results:
            filename = imgs_list[key].split('/')[-1]
            print ('NO. ' + str(count) +' is: ' + filename + ' distance : ' + str(value))
            
            classname = filename.split('_')[0]
            imagetype = filename.split('_')[1][0]
            if imagetype == "i":
                subdir = "/luminence/"
            else:
                subdir = "/rotation/"
            imagename = filename.split('_')[0] + '_' + filename.split('_')[1] + ".png"
            imageaddress = "./min_merged_train/" + classname + subdir + imagename
            print imageaddress
            if count <= 11:
                ranking_image = cv2.imread(imageaddress)
                ax[count].set_title("NO."+ str(count) + " " + imagename.split('.')[0])
                ax[count].set_xlabel(" Dist: " + str(value))
                ranking_image = cv2.cvtColor(ranking_image,cv2.COLOR_BGR2RGB)
                ax[count].imshow(ranking_image)
                ax[count].set_axis_off()           
            if count > 20:
                break
            count += 1
            
        rst_dir = "./image_result/"
        if os.path.exists(rst_dir) == False:
            os.makedirs(rst_dir)         
        plt.savefig(rst_dir + target_addr.split('/')[-1]) 
                                  
    elif (mode == 2): 
        # try to find similar from the database; rank the result; calculat score; store in a csv file
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
        for target, target_filename in img_generator(image_list):
            target_filename = target_filename.split('.')[0]
            target_fileno = target_filename.split('_')[0]
            
            results, imgs_list = searchFromBase(hist_addr, target, kmeans, nfeatures, descriptor_type, mode, has_hist=True)
            
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
                count += 1
                note -= 1
            writer.writerow([str(target_filename), str(first_match), str(second_match), str(note_total)])
            note_global += note_total
            
        writer.writerow(['Conclusion', str(first_count_global), str(second_count_global), str(note_global)])
        csvfile.close()

    # mode for generate results for all types of objects in test set, generate for all class #
    elif mode == 3:
        pr_csv_generation(target_dir, hist_addr, kmeans, nfeatures, descriptor_type, mode)

    # mode for generating result for a given class in the test set #
    elif mode == 4:
        pr_csv_generation(target_dir, hist_addr, kmeans, nfeatures, descriptor_type, mode, class_id)
          
    # mode no exist #
    else:
        print "mode error should be [1~5]"

    # remove hists after usage #
    #shutil.rmtree(hist_addr)


if __name__ == '__main__':
    #run example: python test.py -n 100 -c 50 -d orb -m 1 -i 266_c > ./image_result/266_c.txt #
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", type=int, default=100,
                        help="Number of feature point for each image.")
    parser.add_argument("-c", type=int, default=50,
                        help="Number of cluster for kmeans")
    parser.add_argument("-d", type=str, default='orb',
                        help="Descriptor Type")
    parser.add_argument("-m", type=int, default=1,
                        help="Execution Mode")                                               
    parser.add_argument("--addr", type=str, default='./min_merged_train/',
                        help="training set addr")
    parser.add_argument("--tid", type=str, default=255,
                        help="test image class id for mode 4")
    parser.add_argument("-i", type=str, default="251_i120",
                        help="test image name")
    args = parser.parse_args()
    
    train_addr = args.addr  # './min_merged_train/', path where train images lie
    nfeatures = args.n  # Max quantity of kp, 0 as invalid for brief
    descriptor_type = args.d
    mode = args.m
    class_id = args.tid  # for mode 4, assign class id
    filename = args.i
    
    classname = filename.split('_')[0]
    imagetype = filename.split('_')[1][0]
    if imagetype == "i":
        subdir = "/luminence/"
    else:
        subdir = "/rotation/"
    imagename = filename.split('_')[0] + '_' + filename.split('_')[1] + ".png"
    imageaddress = "./min_merged_test/" + classname + subdir + imagename
    
    print "train_addr : %s, desptype : %s, nfeatures : %d" % (train_addr, descriptor_type, nfeatures)
    print "treating image : " + imageaddress        
    main(train_addr, mode, descriptor_type, nfeatures, class_id, imageaddress)
