# https://www.kaggle.com/datasets/maciejgronczynski/biggest-genderface-recognition-dataset

import numpy as np

from keras.preprocessing.image import ImageDataGenerator

#1. 데이터

train_datagen = ImageDataGenerator(
    rescale=1./255,
)

test_datagen = ImageDataGenerator(
    rescale=1./255
)

path_train="./_data/image/faces/"
# path_test="./_data/image/catdog/test_set/"

xy_train = train_datagen.flow_from_directory(
    directory=path_train,
    target_size=(150,150),
    color_mode="rgb",
    batch_size=30000,
    class_mode="binary",
    shuffle=False

)

# xy_test = test_datagen.flow_from_directory(
#     path_train,
#     target_size=(100,100),
#     color_mode="rgb",
#     batch_size=2500,
#     class_mode="binary",
#     # shuffle=False
# )

# x_train = np.concatenate((xy_train[0][0], xy_train[1][0],xy_train[2][0]),axis=0)
# y_train = np.concatenate((xy_train[0][1], xy_train[1][1],xy_train[2][1]),axis=0)

x_train = xy_train[0][0]
y_train = xy_train[0][1]

# x_test = xy_train[0][0]
# y_test = xy_train[0][1]

np_path = "./_data/npy/faces_npy/"
np.save(np_path+"x_150.npy",x_train)
np.save(np_path+"y_150.npy",y_train)
# np.save(np_path+"x_test.npy",x_test)
# np.save(np_path+"y_test.npy",y_test)

