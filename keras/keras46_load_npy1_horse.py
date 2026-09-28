import time
import datetime

import numpy as np
import pandas as pd

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import AveragePooling2D, Dense, Conv2D, Dropout, Flatten, GlobalAveragePooling2D, MaxPooling2D

from sklearn.metrics import accuracy_score
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

import my_util

path = "./_save/hh/"
date = datetime.datetime.now().strftime("%m%d_%H%M")
prefix = "k46_"+date
filename = "_{epoch:04d}-{val_loss:.4f}.keras"
filepath = "".join([path, prefix, filename])

#1. 데이터

np_path = "./_data/npy/hh_npy/"

x = np.load(np_path+"x_train.npy")
y = np.load(np_path+"y_train.npy")

rand_num = 90
train_ration = 0.8
x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=train_ration, random_state=rand_num, stratify=y)

ohe = OneHotEncoder(sparse_output=False)
y_train = ohe.fit_transform(y_train.reshape(-1,1))
y_test = ohe.transform(y_test.reshape(-1,1))

# print(x_train.shape ,y_train.shape)
# print(x_test.shape ,y_test.shape)

#2. 모델 구성
model = Sequential()
model.add(Conv2D(16, (3,3),input_shape=x_train[0].shape)) # (26,26,64)
model.add(Conv2D(filters=32, kernel_size=(3,3), activation="relu")) #(24, 24, 32)
model.add(MaxPooling2D())
model.add(Dropout(0.1))
model.add(Conv2D(64, (2,2), activation="relu"))
model.add(Dropout(0.2))
model.add(MaxPooling2D())
# model.add(Dropout(0.1))
model.add(Conv2D(32, (2,2), activation="relu"))
model.add(Conv2D(128, (2,2), activation="relu"))
model.add(Dropout(0.3))
model.add(AveragePooling2D())
model.add(Dropout(0.1))
model.add(Conv2D(64, (2,2), activation="relu")) #(20,20,16)
model.add(Conv2D(16, (2,2), activation="relu")) #(20,20,16)
model.add(MaxPooling2D())
model.add(Dropout(0.1))
model.add(Conv2D(32, (3,3))) # (26,26,64)
# model.add(AveragePooling2D())
model.add(Conv2D(128, (3,3),activation="relu")) # (26,26,64)
model.add(Dropout(0.3))

model.add(GlobalAveragePooling2D()) #(None, 6400)
model.add(Dense(units=64, activation="relu"))
# model.add(Dense(units=64, activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(units=32, activation="relu"))
model.add(Dense(2,activation="softmax"))
# model.summary()
# exit()

#3. 컴파일 훈련
model.compile(loss = "categorical_crossentropy", optimizer = "adam", metrics=["acc"])

es = EarlyStopping(
    monitor = 'val_loss', mode = "min", 
    patience = 70, restore_best_weights= True, 
)

mcp = ModelCheckpoint(
    monitor='val_loss', mode='auto', verbose=1,
    save_best_only=True, filepath=filepath,
)

batch_size = 20
start_time = time.time()

history = model.fit(x_train,y_train, verbose=2, epochs=5000, batch_size=batch_size, validation_split=0.25, callbacks = [es,mcp])

train_time = time.time() - start_time

#4. 평가 예측
loss = model.evaluate(x_test,y_test)

y_pred = model.predict(x_test)

y_pred = np.argmax(y_pred,axis=1)
y_test = np.argmax(y_test,axis=1)
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
    csv_file_path="horsehuman.csv"
)

my_util.leaveTop(path=path, prefix=prefix, subfix=".keras", count=5, mode="min")

#target acc 1.0