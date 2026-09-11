import numpy as np
import pandas as pd

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler,MaxAbsScaler
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


import time
import my_util

datasets = fetch_covtype()

x=datasets.data
y=datasets.target

print(x.shape, y.shape) #(178, 13) (178,)
print(np.unique(y, return_counts=True))

y = pd.get_dummies(y)

rand_num = 90
train_ration = 0.8
x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=train_ration, random_state=rand_num, stratify=y)

# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
scaler = RobustScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델
model = Sequential([keras.Input(shape=(54,))])
model.add(Dense(128, activation="swish"))
model.add(Dense(64, activation="relu"))
model.add(Dense(128))
model.add(Dense(128, activation="relu"))
model.add(Dense(32))
model.add(Dense(64, activation="relu"))
model.add(Dense(7, activation="softmax"))


#3. 컴파일 훈련
model.compile(loss = "categorical_crossentropy", optimizer = "adam", metrics=["acc"])

es = EarlyStopping(
    monitor = 'val_loss', mode = "min", 
    patience = 50, restore_best_weights= True, 
)

batch_size = 4096
start_time = time.time()

history = model.fit(x_train,y_train, epochs=30000, batch_size=batch_size, validation_split=0.2, callbacks = [es])

train_time = time.time() - start_time

#4. 평가 예측
loss = model.evaluate(x_test,y_test)

y_pred = model.predict(x_test)
y_pred = np.argmax(y_pred,axis=1)
y_test = np.argmax(y_test,axis=1)
acc = accuracy_score(y_test,y_pred)
print(y_pred)
print(acc)

my_util.record_model_csv(
    model = model,
    data_shape = x_train.shape,
    random_num = rand_num,
    batch_size = batch_size,
    history = history,
    training_time = train_time,
    test_loss = loss,
    r2_score = acc,
    train_ration = train_ration,
    csv_file_path= "covtype.csv"
)


#######acc 93   0.9358880579675224