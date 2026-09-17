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

# path = "./_save/mnist/"
# date = datetime.datetime.now().strftime("%m%d_%H%M")
# prefix = "k41_"+date
# filename = "_{epoch:04d}-{val_loss:.4f}.keras"
# filepath = "".join([path, prefix, filename])

#1. 데이터
(x_train,y_train),(x_test,y_test) = mnist.load_data()
# print(x_train[3])
# print(x_train.shape, y_train.shape) # (60000, 28, 28) (60000,)
# print(x_test.shape, y_test.shape)   # (10000, 28, 28) (10000,)

# print(np.unique(y_train, return_counts=True)) #(array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949],dtype=int64))
# print(pd.value_counts(y_test))
# print(np.max(x_train),np.min(x_train)) #255 0
# print(np.max(x_test),np.min(x_test)) #255 0

# 스케일링 1(Minmax)
x_train = x_train/255.
x_test = x_test/255.

# print(np.max(x_train),np.min(x_train))    #1.0 0.0
# print(np.max(x_test),np.min(x_test))      #1.0 0.0

# 스케일링 2(MaxAbs)
# x_train = (x_train-127.5)/127.5
# x_test = (x_test-127.5)/127.5
# print(np.max(x_train),np.min(x_train))  #1.0 -1.0
# print(np.max(x_test),np.min(x_test))    #1.0 -1.0

x_train = x_train.reshape(-1, 784)
x_test = x_test.reshape(-1, 784)

ohe = OneHotEncoder(sparse_output=False)
y_train = ohe.fit_transform(y_train.reshape(-1,1))
y_test = ohe.transform(y_test.reshape(-1,1))

#2. 모델 구성
model = Sequential()
model.add(Dense(units=256, input_shape=x_train[0].shape))
model.add(Dropout(0.1))
model.add(Dense(units=128, activation="gelu"))
model.add(Dropout(0.1))

model.add(Dense(units=256, activation="gelu"))
model.add(Dropout(0.2))
model.add(Dense(units=64, activation="gelu"))
model.add(Dropout(0.1))

model.add(Dense(units=128, activation="gelu"))
model.add(Dense(units=256, activation="relu"))
model.add(Dropout(0.2))

model.add(Dense(units=256))
model.add(Dropout(0.3))
model.add(Dense(units=128, activation="silu"))

model.add(Dense(units=256, activation="gelu"))
model.add(Dropout(0.1))
model.add(Dense(units=256))

model.add(Dense(units=64, ))
model.add(Dropout(0.3))
model.add(Dense(units=32))

model.add(Dense(10, activation="softmax"))

#3. 컴파일 훈련
model.compile(loss = "categorical_crossentropy", optimizer = "adam", metrics=["acc"])

es = EarlyStopping(
    monitor = 'val_loss', mode = "min", 
    patience = 30, restore_best_weights= True, 
)

# mcp = ModelCheckpoint(
#     monitor='val_loss', mode='auto', verbose=1,
#     save_best_only=True, filepath=filepath,
# )

batch_size = 64
start_time = time.time()

history = model.fit(x_train,y_train, verbose=2, epochs=500, batch_size=batch_size, validation_split=0.3, callbacks = [es])#,mcp])

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

# my_util.leaveTop(path=path, prefix=prefix, subfix=".keras", count=5, mode="min")