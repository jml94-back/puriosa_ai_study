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

train_datagen = ImageDataGenerator(
    #증폭 변환하는 파라미터들
    rescale=1./255,
    # horizontal_flip=True,   # 수평 반전
    # vertical_flip=True,     # 수직 반전
    # width_shift_range= 0.1, # 평형 이동
    # height_shift_range=0.1, # 수직 이동
    # rotation_range= 5,      # 각도 조절
    # zoom_range=1.2,         #
    # shear_range=0.7,        # 좌표 하나를 고정하고 다른 좌표들을 이동
    # fill_mode="nearest",
)

test_datagen = ImageDataGenerator(
    rescale=1./255
)

path_train="./_data/image/brain/train/"
path_test="./_data/image/brain/test/"

xy_train = train_datagen.flow_from_directory(
    directory=path_train,
    target_size=(150,150),
    color_mode="grayscale",
    batch_size=160,
    class_mode="binary",
    shuffle=True

)

xy_test = test_datagen.flow_from_directory(
    path_test,
    target_size=(150,150),
    color_mode="grayscale",
    batch_size=120,
    class_mode="binary",
    # shuffle=False
)

x_train = xy_train[0][0]
y_train = xy_train[0][1]

x_test = xy_test[0][0]
y_test = xy_test[0][1]

np_path = "./_data/brain_npy/"
np.save(np_path+"x_train.npy",x_train)
np.save(np_path+"y_train.npy",y_train)
np.save(np_path+"x_test.npy",x_test)
np.save(np_path+"y_test.npy",y_test)

