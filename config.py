#!/usr/bin/env python
"""
Set up environment and config variables
"""

import os
import pandas as pd
import numpy as np

# These are the locations of the images provided by Kaggle
# Root Dir is needed for Python, but not for create lmdb shell script later... (we need it there too!)
image_root_dir = './imgs/'
train_image_source_dir = "./train/"
test_image_source_dir = "./test/"
driver_image_list = "./driver_imgs_list.csv"

# These are the locations of the images that we will work with
# Note that as we're continually mix up training and validation drivers/images,
# then we will store images in one directory and use code to determine whether to train or validate
train_images_dir = "./images/train/"
#validation_images_dir = "./images/validate/"
test_images_dir = "./images/test/"

# Some more controls
# color type: 1 - grey, 3 - rgb
color_type = 1
image_width = 224
image_height = 224


# Training set is in the provided csv file
def get_driver_list():
    return pd.read_csv(driver_image_list)

# Get a list of all the test images
def get_test_image_list():
    test_images_list = os.listdir(image_root_dir + test_image_source_dir)
    print "Total number of test images found {}".format(len(test_images_list))
    return test_images_list


# Create a list of images and classes for the training set
# images, classes = get_driver_images_and_classes(driver_list)
def get_driver_images_and_classes(driver_list=get_driver_list()):
    image_list = []
    class_list = []
    total = 0
    for driver_row in [ drvr for drvr in driver_list.iterrows() ]:   # if drvr[1]['subject'] in filter
        driver = driver_row[1]  # Drop the index created by the Pandas Dataframe
        driver_class = int(driver['classname'][1:])  # Get integer to represent class (eg 'c0' is class '0')
        image_list.append(driver['img'])
        class_list.append(driver_class)
        total += 1
    print "Total number of training images found {}".format(total)
    #Return a list of images and their classification
    return np.array(image_list), np.array(class_list)

