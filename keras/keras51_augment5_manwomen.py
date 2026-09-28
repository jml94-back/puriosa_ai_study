# https://www.kaggle.com/datasets/maciejgronczynski/biggest-genderface-recognition-dataset
import time
import datetime

import numpy as np
import pandas as pd

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import AveragePooling2D, Dense, Conv2D, Dropout, Flatten, GlobalAveragePooling2D, MaxPooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint
from tqdm_callback import TQDMProgress

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import my_util

path = "./_save/manwoman/"
date = datetime.datetime.now().strftime("%m%d_%H%M")
prefix = "k51_"+date
filename = "_{epoch:04d}-{val_loss:.4f}.keras"
filepath = "".join([path, prefix, filename])

#1. 데이터

np_path = "./_data/npy/faces_npy/"
x = np.load(np_path+"x.npy")
y = np.load(np_path+"y.npy")

rand_num = 90
train_ration = 0.01
x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=train_ration, random_state=rand_num, stratify=y)

# print(y_train[0:50])
# print(np.unique(y_train, return_counts=True)) #14142,  7591
# exit()

x_train_woman = x_train[np.where(y_train>0.0)]

datagen = ImageDataGenerator(
    #증폭 변환하는 파라미터들
    # rescale=1./255,
    horizontal_flip=True,   # 좌우 반전
    # vertical_flip=True,     # 상하 반전
    width_shift_range= 0.1, # 평형 이동
    height_shift_range=0.1, # 수직 이동
    # rotation_range= 20,      # 각도 조절
    zoom_range=0.1,         #
    # shear_range=0.7,        # 좌표 하나를 고정하고 다른 좌표들을 이동
    fill_mode="nearest",
)

augment_size = 6551 # 14142 - 7591 데이터 갯수 차이만큼 증폭

randidx = np.random.choice(x_train_woman.shape[0],size=augment_size)

x_augmented = x_train_woman[randidx].copy()
y_augmented = np.ones(augment_size)
x_augmented = datagen.flow(
    x_augmented,
    y_augmented,
    batch_size=augment_size,
    shuffle=False,
).next()[0]

x_train = np.concatenate((x_train,x_augmented))
y_train = np.concatenate((y_train,y_augmented))
# print(x_train.shape ,y_train.shape)
# print(x_test.shape ,y_test.shape)
# print(np.unique(y_train, return_counts=True))
# exit()

#2. 모델 구성
model = Sequential()
model.add(Conv2D(16, (3,3),input_shape=x_test[0].shape)) # (26,26,64)
model.add(Conv2D(filters=32, kernel_size=(3,3), activation="relu")) #(24, 24, 32)
model.add(Dropout(0.2))
model.add(MaxPooling2D())
model.add(Dropout(0.1))
model.add(Conv2D(32, (2,2), activation="relu"))
model.add(Conv2D(128, (2,2), activation="relu"))
model.add(Dropout(0.3))
model.add(MaxPooling2D())
model.add(Dropout(0.1))
model.add(Conv2D(32, (3,3),input_shape=x_test[0].shape)) # (26,26,64)
model.add(Conv2D(128, (3,3),input_shape=x_test[0].shape)) # (26,26,64)
model.add(Dropout(0.3))

model.add(GlobalAveragePooling2D()) #(None, 6400)
model.add(Dense(units=64, activation="relu"))
# model.add(Dense(units=64, activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(units=32, activation="relu"))
model.add(Dense(1,activation="sigmoid"))
model.summary()
# exit()

#3. 컴파일 훈련
model.compile(loss = "binary_crossentropy", optimizer = "adam", metrics=["acc"])

es = EarlyStopping(
    monitor = 'val_loss', mode = "min", 
    patience = 30, restore_best_weights= True, 
)

mcp = ModelCheckpoint(
    monitor='val_loss', mode='auto', verbose=0,
    save_best_only=True, filepath=filepath,
)

tqdm_callback = TQDMProgress()

batch_size = 200
start_time = time.time()

history = model.fit(x_train,y_train, verbose=0, epochs=5000, batch_size=batch_size, validation_split=0.3, callbacks = [es,mcp,tqdm_callback])

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
    random_num = rand_num,
    batch_size = batch_size,
    history = history,
    training_time = train_time,
    test_loss = loss[0],
    sub_score = acc,
    train_ration = train_ration,
    csv_file_path="manwoman.csv"
)

my_util.leaveTop(path=path, prefix=prefix, subfix=".keras", count=5, mode="min")