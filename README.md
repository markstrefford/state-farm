# state-farm

Kaggle State Farm competition, used for Udacity Machine Learning Nanodegree Capstone

## Overview

This project aims to determine whether it is possible to identify what action a driver is taking from a static image, 
for example are they driving safely or distracted by texting, phoning, drinking, talking to a passenger, etc.

See https://www.kaggle.com/c/state-farm-distracted-driver-detection for more details on the competition

## Installation

This project is based on Python 2.7 with the following plugins / frameworks

1. pandas
1. numpy
1. matplotlib
1. pydot-ng
1. graphviz
1. Theano
1. Keras
1. OpenCV v2.4.13 http://opencv.org

## Running the model

### Data Files
Download the data files from 
https://www.kaggle.com/c/state-farm-distracted-driver-detection/data 
and extract into the root folder of this project


### Using the Python Notebook
All code for this project is included in the iPython notebook state-farm.ipynb. 

To use the CPU only:
ipython notebook

To use a GPU:
THEANO_FLAGS=device=gpu,floatX=float32 ipython notebook


### Testing and submitting data to Kaggle
Predictions are written to time-stamped files in the submission folder.  These can be uploaded 
at this URL:

https://www.kaggle.com/c/state-farm-distracted-driver-detection/submit


## Classes of data

The 10 classes to predict are:

c0: safe driving
c1: texting - right
c2: talking on the phone - right
c3: texting - left
c4: talking on the phone - left
c5: operating the radio
c6: drinking
c7: reaching behind
c8: hair and makeup
c9: talking to passenger




