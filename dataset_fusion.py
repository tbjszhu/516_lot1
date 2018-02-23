#!/usr/bin/python
# -*- coding: UTF-8 -*-
# this code is for merging two datasets into one #
import glob
import os
from shutil import copyfile


### definition ###
train_1_path = "./train_rot/" # where to save train rotated images
test_1_path = "./test_rot/" # where to save test rotated images
train_2_path = "./train_col/" # where to save train rotated images of luminence change
test_2_path = "./test_col/" # where to save test images of luminence change

merged_train = "./merged_train/" # final train set
merged_test = "./merged_test/" # final test set

# create train directory
if not os.path.isdir(merged_train):
    os.makedirs(merged_train[0:-1])
    print 'create ' + merged_train


# create test directory
if not os.path.isdir(merged_test):
    os.makedirs(merged_test[0:-1])
    print 'create ' + merged_test


# put rotated images into merged_train

fileaddr = glob.glob(train_1_path+'*.png')
print fileaddr
for file in fileaddr:
    filename = file.split('/')[-1]
    print filename
    cls = filename.split('_')[0] # get class from filename
    print cls

    if not os.path.isdir(merged_train+cls+'/'+'rotation'):
        os.makedirs(merged_train+cls+'/'+'rotation')
        print 'create ' + merged_train+cls+'/'+'rotation'

    copyfile(file, merged_train+cls+'/'+'rotation/'+filename)


# put rotated images into merged_test
fileaddr = glob.glob(test_1_path+'*.png')
print fileaddr
for file in fileaddr:
    filename = file.split('/')[-1]
    print filename
    cls = filename.split('_')[0]  # get class from filename
    print cls

    if not os.path.isdir(merged_test+cls+'/'+'rotation'):
        os.makedirs(merged_test+cls+'/'+'rotation')
        print 'create ' + merged_test+cls+'/'+'rotation'

    copyfile(file, merged_test+cls+'/'+'rotation/'+filename)



# put images of luminence change into merged_train

fileaddr = glob.glob(train_2_path+'*.png')
print fileaddr
for file in fileaddr:
    filename = file.split('/')[-1]
    print filename
    cls = filename.split('_')[0]  # get class from filename
    print cls

    if not os.path.isdir(merged_train+cls+'/'+'luminence'):
        os.makedirs(merged_train+cls+'/'+'luminence')
        print 'create ' + merged_train+cls+'/'+'luminence'

    copyfile(file, merged_train+cls+'/'+'luminence/'+filename)

# put images of luminence change into merged_test
fileaddr = glob.glob(test_2_path+'*.png')
print fileaddr
for file in fileaddr:
    filename = file.split('/')[-1]
    print filename
    cls = filename.split('_')[0]  # get class from filename
    print cls

    if not os.path.isdir(merged_test+cls+'/'+'luminence'):
        os.makedirs(merged_test+cls+'/'+'luminence')
        print 'create ' + merged_test+cls+'/'+'luminence'

    copyfile(file, merged_test+cls+'/'+'luminence/'+filename)







