import numpy as np
import pandas as pd

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint

from sklearn.preprocessing import MinMaxScaler,StandardScaler,MaxAbsScaler,RobustScaler
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import time
import datetime
import my_util

path = "./_save/digits/"
date = datetime.datetime.now().strftime("%m%d_%H%M")
filename = "_{epoch:04d}-{val_loss:.4f}.keras"
filepath = "".join([path,"k31_",date, filename])

#1. 데이터

datasets = load_digits()

# print(datasets.DESCR)

x=datasets.data
y=datasets.target

# print(x.shape, y.shape) #(1797, 64) (1797,)
# print(np.unique(y,return_counts=True)) #(array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), array([178, 182, 177, 183, 181, 182, 181, 179, 174, 180]))
# print(pd.isna(x).sum()) #결측치 없음

y = pd.get_dummies(y, dtype=int)


rand_num = 8765
train_ratio=0.8

x_train,x_test,y_train,y_test = train_test_split(x,y, train_size= train_ratio, random_state=rand_num,stratify=y)

# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
scaler = RobustScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델
model = Sequential([keras.Input(shape=(64,))])
model.add(Dense(256))
model.add(Dropout(0.3))
model.add(Dense(128, activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(256, activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(128))
model.add(Dropout(0.2))
model.add(Dense(256, activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(128, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(256, activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(64))
model.add(Dense(10, activation="softmax"))

#3. 컴파일 훈련
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["acc"])

es = EarlyStopping(
    monitor='val_loss', mode="min",
    patience= 100, restore_best_weights= True,
)

mcp = ModelCheckpoint(
    monitor='val_loss', mode='auto', verbose=1,
    save_best_only=True, filepath=filepath,
)

batch_size=16
start_time = time.time()

history = model.fit(x_train,y_train, epochs=100000, batch_size = batch_size,validation_split=0.2,callbacks=[es,mcp])
train_time=time.time()-start_time
#4. 평가 예측

loss = model.evaluate(x_test,y_test)

y_pred = model.predict(x_test)
y_pred = np.argmax(y_pred,axis=1)
y_test = np.argmax(y_test,axis=1)
acc = accuracy_score(y_test,y_pred)
# print(y_pred)
# print(acc)

my_util.record_model_csv(
    model = model,
    data_shape = x_train.shape,
    random_num = rand_num,
    batch_size = batch_size,
    history = history,
    training_time = train_time,
    test_loss = loss[0],
    r2_score = acc,
    train_ration = train_ratio,
    csv_file_path="digit.csv"
)