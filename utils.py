import numpy as np
import cv2
from matplotlib import pyplot as plt
import glob
import os
from itertools import groupby
import sys
from collections import OrderedDict
import random
import csv

# Read images/masks from a directory
ranking_width = 20


def getImageListFromDir(img_dir, filetype='png'):
    """
    :param img_dir: imgs root dir, string
    :param filetype: "png", "jpg" or "bmp", string
    :return: list of images, list
    """
    img_dir = img_dir + '/*/*/' + '*.' + filetype
    l = sorted(glob.glob(img_dir))  # ./merged_train/999/rotation/999_r.png
    return l


def getHistListFromDir(img_dir):
    """
    :param img_dir: imgs root dir, string
    :param filetype: "png", "jpg" or "bmp", string
    :return: list of images, list
    """
    filetype = 'npy'
    img_dir = img_dir + '*.' + filetype
    l = sorted(glob.glob(img_dir))  # ./merged_train/999/rotation/999_r.png
    return l


# read images with img generator

def img_generator(img_list):
    """
    :param img_list: list of iamges, list of string
    :return: yield a pair of sequences images
    """
    while len(img_list) > 0:
        f1 = img_list.pop(0)
        print "read img: ", f1.split('/')[-1]
        img = cv2.imread(f1)
        yield (img, f1.split('/')[-1])


# calculate ORB descriptors
def orb_descriptor_generator(data, feature_point_quantity):
    """
    :param data: numpy array grayscale image for getting histo or local descriptors for an images
    :param feature_point_quantity: MAX feature point quantity
    :return: keypoint_list, descriptor_list
    """
    orb = []
    kp = []

    if cv2.__version__[0] == '3':
        orb = cv2.ORB_create(nfeatures=feature_point_quantity)
    else:
        orb = cv2.ORB(nfeatures=feature_point_quantity)

    # find the keypoints with ORB
    kp = orb.detect(data, None)

    # compute the descriptors with ORB
    kp, des = orb.compute(data, kp)
    return (kp, des)


def brief_descriptor_generator(data, nfeatures):
    """
    :param data: numpy array grayscale image for getting histo or local descriptors for an images
    :param feature_point_quantity: MAX feature point quantity
    :return: keypoint_list, descriptor_list
    """

    # nfeatures can not be controlled in STAR
    # Initiate STAR detector
    star = []
    brief = []
    if cv2.__version__[0] == "2":
        star = cv2.FeatureDetector_create("STAR")

        # Initiate BRIEF extractor
        brief = cv2.DescriptorExtractor_create("BRIEF")

    else:
        star = cv2.xfeatures2d.StarDetector_create()
        brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()

    # find the keypoints with STAR
    kp = star.detect(data, None)

    # compute the descriptors with BRIEF
    kp, des = brief.compute(data, kp)
    if des is not None :
        print "des shape", des.shape

    return kp, des


def SIFT_descriptor_generator(data, nfeatures):
    """
    :param data: numpy array grayscale image for getting histo or local descriptors for an images
    :param feature_point_quantity: MAX feature point quantity
    :return: keypoint_list, descriptor_list
    """
    """
    gray = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)

    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray, 2, 3, 0.04)
    
    """
    sift = cv2.xfeatures2d.SIFT_create(nfeatures=nfeatures)
    kps, descs = sift.detectAndCompute(data,None)
    if descs is not None:
        print "des shape", descs.shape
    return kps, descs

# generator descriptors
def generator_descriptor(fileaddr, save_addr, nfeatures, desp_type):
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
    if fileaddr[-1] != '/':
        fileaddr += '/'
    # fileList = os.listdir(fileaddr)
    fileList = glob.glob(fileaddr + "*/*/*.png")
    print fileList
    for file in fileList:
        if '.png' in file:
            print file
            filename_des = file.split('/')[-1].split('.')[0]
            data = cv2.imread(file, 0)

            # todo: add new detectors here !

            if desp_type == "orb":
                kp, des = orb_descriptor_generator(data, nfeatures)

            elif desp_type == "brief":
                kp, des = brief_descriptor_generator(data, nfeatures)

            elif desp_type == "sift":
                kps, des = SIFT_descriptor_generator(data, nfeatures)
            else:
                print "Algo : " + desp_type + " is not supported"

            # creat sub dir classified by nfeatures to store descriptor
            subdir_addr = save_addr + '/' + 'nf_' + str(nfeatures)
            if not os.path.isdir(subdir_addr):
                os.makedirs(subdir_addr)
                print 'create ' + subdir_addr
            # creat file to store descriptor
            np.save(subdir_addr + '/' + filename_des + '_' + desp_type, des)
            # np.save(save_addr+'/'+filename_des+'_kp', kp)


# generator hists (global descriptors) from data (local descriptors or image)
def generateHist(model, data, data_type, nfeatures, decpt_type):
    """
    :param model: k-means model trained by descriptors in database.
    :param data: numpy array grayscale image for getting histo or local descriptors for an images
    :param data_type : string, "dscpt" or "image"
    :param decpt_type : string , type of descriptor, e.g. "orb"
    :return: an histogramme for the given data
    """

    if data_type == "dscpt":
        # from k-means prediction to histogram (global descriptor) (#bins = #classes)
        res = np.zeros((1, model.get_params()['n_clusters']), dtype=np.float32)
        if data is None:  # if orb cannot get any keypoints
            return res

        label = model.predict(data)
        for value in label:
            res[0, value] += 1.0

        return res / np.sum(res)  # normalized histogram

    elif data_type == "image":

        if decpt_type == "orb":
            kp, des = orb_descriptor_generator(data, nfeatures)

        elif decpt_type == "brief":
            kp, des = brief_descriptor_generator(data, nfeatures)

        elif decpt_type == "sift":
            kp, des = SIFT_descriptor_generator(data, nfeatures)
        else:
            print "Algo : " + decpt_type + " is not supported"

        # from k-means prediction to histogram (#bins = #classes)
        res = np.zeros((1, model.get_params()['n_clusters']), dtype=np.float32)
        if des is None:  # if orb cannot get any keypoints
            return res
        des = np.asarray(des, dtype=np.float32)
        #des_float = []
        #for i in range(len(des)):  # convert des from int to float to avoid type warning
        #    des_float.append(map(float, des[i]))
        label = model.predict(des)
        for value in label:
            res[0, value] += 1.0
        return res / np.sum(res)  # normalized histogram

    else:
        print ("data type error")
        return 0


# search similar images of a given target (gs image) in terms of distance of hists.

def searchFromBase(base_dir, target, model, nfeatures, descriptor_type, has_hist):
    """
    :param base_dir: search base of images
    :param target: target image numpy grayscale image
    :param model : kmeans pretrained model
    :param has_hist : bool, if true base_dir is the addr for the hists, else it is the addr for images
    :return: a list of ranking of top 10 similars images, [ (index, distance value)] and a list of image dir
    """
    if has_hist:
        imgs_addr = getHistListFromDir(base_dir)
    else:
        imgs_addr = getImageListFromDir(base_dir)
    dist = {}
    target_hist = generateHist(model, target, 'image', nfeatures, descriptor_type).astype(np.float32)
    # print np.sum(target_hist)

    # calculate distance between target hist and base hists
    for idx, img_addr in enumerate(imgs_addr):
        img_gs = []
        hist = []
        if has_hist == False:
            print img_addr
            img_gs = cv2.imread(img_addr, '0')
            hist = generateHist(model, img_gs, 'image', descriptor_type)
        else:
            hist = np.load(img_addr)
        dist[idx] = np.linalg.norm(hist - target_hist)  # eucudian distance

    # get the top 10 ranking
    sorted_d = OrderedDict(sorted(dist.items(), key=lambda x: x[1]))
    dictlist = []
    for key, value in sorted_d.items():
        temp = [key, value]  # [index, distance]
        dictlist.append(temp)
    return dictlist[0:ranking_width], imgs_addr


def get_class_image_list(target_dir, class_name):
    l = glob.glob(target_dir + '/' + class_name + '/*/*')
    return l


def generate_random_image_list(image_list, class_name, class_start, class_num, num):
    class_name = str(class_name)
    image_list_temp = image_list[:]  # to store different image classes
    same_class_image_count = 0
    same_class_list = []

    for item in image_list_temp:  # find the files from the same class
        if str(class_name) in item:
            same_class_list.append(item)
            same_class_image_count += 1
    for item in same_class_list:  # delete the files from the same class
        image_list_temp.remove(item)

    rand_file_num_list = []
    while True:  # generate "num" defferent file num
        rand_file_num = str(random.randint(class_start, class_start + (class_num - 1) * same_class_image_count - 1))
        if (rand_file_num not in rand_file_num_list):
            rand_file_num_list.append(rand_file_num)
        else:
            continue
        if len(rand_file_num_list) == num:
            break;
    rand_image_list = []
    for item in rand_file_num_list:
        rand_image_list.append(image_list_temp[int(item) - class_start])

    return rand_image_list


def csv_init(csv_file_path, kmeans, nfeatures, class_name, descriptor_type):
    csv_file_name = csv_file_path + '/kmeans_' + str(kmeans.get_params()['n_clusters']) + '_nf_' + str(
        nfeatures) + descriptor_type + '_class_' + class_name + '.csv'
    if os.path.exists(csv_file_path) == False:
        os.mkdir(csv_file_path)
    csvfile = file(csv_file_name, 'wb')
    index = range(ranking_width)
    index_str = map(str, index)
    file_header = ['id'] + index_str + ['Total']
    writer = csv.writer(csvfile)
    writer.writerow(file_header)
    return csvfile, writer


def csv_deinit(csvfile, writer, score_global):
    writer.writerow(['Conclusion'] + score_global)
    csvfile.close()


def pr_csv_generation(target_dir, sub_hist_addr, kmeans, nfeatures, descriptor_type, class_id=-1, has_hist=True):
    image_list = getImageListFromDir(target_dir)
    class_list = []
    dir_list = glob.glob(target_dir + '/*')
    for item in dir_list:
        class_list.append(item.split('/')[-1])
    class_num = len(class_list)
    class_start = int(class_list[0])

    if class_num <= 0:
        print "class_num 0 error"
        sys.exit(0)
    if class_id != -1:
        class_id = str(class_id)
        if class_id in class_list:
            class_list = [class_id]
        else:
            print "class_id: %d not in class_list" % (class_id)

    for class_name in class_list:  # iteration for each class
        class_image_list = get_class_image_list(target_dir, class_name)
        random_image_list = generate_random_image_list(image_list, class_name, class_start, class_num, 5)
        class_image_list.extend(random_image_list)  # joint two lists together

        csv_file_path = './pr_csv'
        csvfile, writer = csv_init(csv_file_path, kmeans, nfeatures, class_name, descriptor_type)
        score_global = [0] * (ranking_width + 1)

        for target, target_filename in img_generator(
                class_image_list):  # iteration for each test image from this class

            target_filename = target_filename.split('.')[0]
            target_class = target_filename.split('_')[0]

            score_vector = [0] * ranking_width
            score_total = 0
            results, imgs_list = searchFromBase(sub_hist_addr, target, kmeans, nfeatures, descriptor_type,
                                                has_hist=True)
            count = 0
            for key, value in results:
                if value == 0:
                    print "error: value in denominator is 0"
                    sys.exit(0)
                score = 1.0 / value
                filename = imgs_list[key].split('/')[-1]
                matched_class = filename.split('_')[0]
                if matched_class == class_name:
                    score_vector[count] = score
                    score_global[count] += score
                    score_total += score
                count += 1
            score_vector_str = map(str, score_vector)
            writer.writerow([str(target_filename)] + score_vector + [str(score_total)])
            score_global[-1] += score_total
        score_global_str = map(str, score_global)
        csv_deinit(csvfile, writer, score_global_str)