import time
import datetime

import numpy as np
import pandas as pd

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import AveragePooling2D, Dense, Conv2D, Dropout, Flatten, GlobalAveragePooling2D, GlobalAvgPool2D, MaxPooling2D

from sklearn.metrics import accuracy_score
from keras.callbacks import EarlyStopping

import my_util

#1. 데이터
np_path = "./_data/catdog_npy/"
x_train = np.load(np_path+"x_train.npy")
y_train = np.load(np_path+"y_train.npy")
x_test = np.load(np_path+"x_test.npy")
y_test = np.load(np_path+"y_test.npy")
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)
