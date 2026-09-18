import numpy as np

from keras.preprocessing.image import ImageDataGenerator

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

path_train="./_data/image/catdog/training_set/"
path_test="./_data/image/catdog/test_set/"

xy_train = train_datagen.flow_from_directory(
    directory=path_train,
    target_size=(100,100),
    color_mode="rgb",
    batch_size=3000,
    class_mode="binary",
    shuffle=True

)

xy_test = test_datagen.flow_from_directory(
    path_test,
    target_size=(100,100),
    color_mode="rgb",
    batch_size=2500,
    class_mode="binary",
    # shuffle=False
)

x_train = np.concatenate((xy_train[0][0], xy_train[1][0],xy_train[2][0]),axis=0)
y_train = np.concatenate((xy_train[0][1], xy_train[1][1],xy_train[2][1]),axis=0)

x_test = xy_test[0][0]
y_test = xy_test[0][1]

np_path = "./_data/npy/catdog_npy/"
np.save(np_path+"x_train_100.npy",x_train)
np.save(np_path+"y_train_100.npy",y_train)
np.save(np_path+"x_test_100.npy",x_test)
np.save(np_path+"y_test_100.npy",y_test)

