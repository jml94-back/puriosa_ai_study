import time
import datetime

import numpy as np
import pandas as pd

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D, GlobalAveragePooling2D
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from keras.callbacks import EarlyStopping, ModelCheckpoint

import my_util

path = "./_save/mnist/"
date = datetime.datetime.now().strftime("%m%d_%H%M")
prefix = "k40_"+date
filename = "_{epoch:04d}-{val_loss:.4f}.keras"
filepath = "".join([path, prefix, filename])

#1. 데이터
(x_train,y_train),(x_test,y_test) = mnist.load_data()

# 스케일링 1(Minmax)
x_train = x_train/255.
x_test = x_test/255.

x_train = x_train.reshape(-1,28,28,1)
x_test = x_test.reshape(-1,28,28,1)

ohe = OneHotEncoder(sparse_output=False)
y_train = ohe.fit_transform(y_train.reshape(-1,1))
y_test = ohe.transform(y_test.reshape(-1,1))

#2. 모델 구성
model = Sequential()
model.add(Conv2D(64, (3,3),input_shape=x_test[0].shape)) # (26,26,64)
model.add(Conv2D(filters=32, kernel_size=(3,3), activation="relu")) #(24, 24, 32)
model.add(Dropout(0.2))
model.add(Conv2D(32, (2,2), activation="relu"))
model.add(MaxPooling2D())
model.add(Conv2D(16, (4,4), activation="relu"))
model.add(Dropout(0.2))
model.add(Conv2D(16, (2,2), activation="relu"))
model.add(Dropout(0.2))
model.add(Conv2D(16, (2,2), activation="relu")) #(20,20,16)

model.add(GlobalAveragePooling2D()) #(None, 6400)
model.add(Dense(units=128, activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(units=32, activation="relu"))
model.add(Dense(10, activation="softmax"))

#3. 컴파일 훈련
model.compile(loss = "categorical_crossentropy", optimizer = "adam", metrics=["acc"])

es = EarlyStopping(
    monitor = 'val_loss', mode = "min", 
    patience = 30, restore_best_weights= True, 
)

mcp = ModelCheckpoint(
    monitor='val_loss', mode='auto', verbose=1,
    save_best_only=True, filepath=filepath,
)

batch_size = 64
start_time = time.time()

history = model.fit(x_train,y_train, verbose=2, epochs=500, batch_size=batch_size, validation_split=0.3, callbacks = [es,mcp])

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
    random_num = 0,
    batch_size = batch_size,
    history = history,
    training_time = train_time,
    test_loss = loss[0],
    sub_score = acc,
    train_ration = 0,
    csv_file_path="mnist.csv"
)
#model, history 등등을 넘기면 csv에 학습한 모델과 loss 등등을 기록해줌
#모델 구성을 기록해줘서 좋습니다.


my_util.leaveTop(path=path, prefix=prefix, subfix=".keras", count=5, mode="min")
#path 폴더에 있는 prefix로 시작하고, subfix로 끝나는 파일 중에
#mode(min,max)에 따라 count개 만큼만 남김 