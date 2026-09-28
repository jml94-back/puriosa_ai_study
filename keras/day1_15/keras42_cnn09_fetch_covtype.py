import numpy as np
import pandas as pd

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler,MaxAbsScaler
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


import time
import datetime
import my_util

# path = "./_save/fetch_covtype/"
# date = datetime.datetime.now().strftime("%m%d_%H%M")
# prefix = "k42_"+date
# filename = "_{epoch:04d}-{val_loss:.4f}.keras"
# filepath = "".join([path, prefix, filename])

#1. 데이터
datasets = fetch_covtype()

x=datasets.data
y=datasets.target

y = pd.get_dummies(y)

rand_num = 90
train_ration = 0.8
x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=train_ration, random_state=rand_num, stratify=y)

scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

x_train = np.array(x_train).reshape(-1,6,9,1)
x_test = np.array(x_test).reshape(-1,6,9,1)

#2. 모델
model = Sequential()
model.add(Conv2D(32, (2,2), input_shape=x_test[0].shape)) # (26,26,64)
model.add(Conv2D(filters=32, kernel_size=(2,2), activation="relu")) #(24, 24, 32)
model.add(Dropout(0.2))

model.add(Flatten()) #(None, 6400)
model.add(Dense(units=128, activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(units=64, activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(units=128, activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(units=128, activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(units=32, activation="relu"))
model.add(Dense(7, activation="softmax"))


#3. 컴파일 훈련
model.compile(loss = "categorical_crossentropy", optimizer = "adam", metrics=["acc"])

es = EarlyStopping(
    monitor = 'val_loss', mode = "min", 
    patience = 50, restore_best_weights= True, 
)

# mcp = ModelCheckpoint(
#     monitor='val_loss', mode='auto', verbose=1,
#     save_best_only=True, filepath=filepath,
# )

batch_size = 4096
start_time = time.time()

history = model.fit(x_train,y_train, epochs=30000, batch_size=batch_size, validation_split=0.2, callbacks = [es])#,mcp])

train_time = time.time() - start_time

#4. 평가 예측
loss = model.evaluate(x_test,y_test)

y_pred = model.predict(x_test)
y_pred = np.argmax(y_pred,axis=1)
y_test = np.argmax(y_test,axis=1)
acc = accuracy_score(y_test,y_pred)

my_util.record_model_csv(
    model = model,
    data_shape = x_train.shape,
    random_num = rand_num,
    batch_size = batch_size,
    history = history,
    training_time = train_time,
    test_loss = loss,
    sub_score = acc,
    train_ration = train_ration,
    csv_file_path= "covtype.csv"
)

# CPU 793.0594 epoch 913
# GPU 190.191  epoch 664

#######acc 93   0.9358880579675224