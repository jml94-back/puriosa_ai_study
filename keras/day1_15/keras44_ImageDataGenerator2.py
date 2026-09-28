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

# print(x_train.shape ,y_train.shape)
# print(x_test.shape ,y_test.shape)

#2. 모델 구성
model = Sequential()
model.add(Conv2D(16, (3,3),input_shape=x_test[0].shape)) # (26,26,64)
model.add(Conv2D(filters=32, kernel_size=(3,3), activation="relu")) #(24, 24, 32)
# model.add(Dropout(0.1))
model.add(Conv2D(32, (2,2), activation="relu"))
model.add(Dropout(0.2))
model.add(MaxPooling2D())
model.add(Dropout(0.1))
model.add(Conv2D(32, (2,2), activation="relu"))
# model.add(Dropout(0.3))
model.add(Conv2D(16, (2,2), activation="relu")) #(20,20,16)


model.add(Flatten()) #(None, 6400)
model.add(Dense(units=256, activation="relu"))
# model.add(Dense(units=64, activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(units=32, activation="relu"))
model.add(Dense(1,activation="sigmoid"))
# model.summary()
# exit()

#3. 컴파일 훈련
model.compile(loss = "binary_crossentropy", optimizer = "adam", metrics=["acc"])

es = EarlyStopping(
    monitor = 'val_loss', mode = "min", 
    patience = 100, restore_best_weights= True, 
)

batch_size = 16
start_time = time.time()

history = model.fit(x_train,y_train, verbose=2, epochs=5000, batch_size=batch_size, validation_split=0.25, callbacks = [es]) #,mcp])

train_time = time.time() - start_time

#4. 평가 예측
loss = model.evaluate(x_test,y_test)

y_pred = model.predict(x_test)
y_pred = np.round(y_pred) #accuracy_score 에서 필요
# print(y_pred[:10])

acc = accuracy_score(y_test,y_pred)
print(acc)

my_util.record_model_csv(
    model = model,
    data_shape = x_train.shape,
    random_num = 0,
    batch_size = batch_size,
    history = history,
    training_time = train_time,
    test_loss = loss[0],
    sub_score = acc,
    train_ration = 0,
    csv_file_path="brain.csv"
)