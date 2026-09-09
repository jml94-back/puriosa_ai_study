import numpy as np
import pandas as pd

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import time
import my_util

#1. 데이터
datasets = load_iris()
# print(datasets.DESCR)

x = datasets.data
y = datasets.target

# one hot encoding
# 0 부터 시작
# from tensorflow.keras.utils import to_categorical
# y = to_categorical(y)

# pandas
# y = pd.get_dummies(y, dtype=int)

# sklearn
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(sparse_output=False) #sparse_output 끄기
y = encoder.fit_transform(y.reshape(-1, 1)) #reshape는 내용,순서가 바뀌면 안됨

print(y)
exit()
# print(x.shape, y.shape)  #(150, 4) (150,)
# print(y)
# exit()
# print(np.unique(y, return_counts=True)) #(array([0, 1, 2]), array([50, 50, 50]))
rand_num = 425
train_ration = 0.8
x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=train_ration, random_state=rand_num, stratify=y)

#2. 모델
model = Sequential([keras.Input(shape=(4,))])
model.add(Dense(16, activation="relu"))
model.add(Dense(32, activation="relu"))
model.add(Dense(128, activation="relu"))
model.add(Dense(128, activation="relu"))
model.add(Dense(3, activation="softmax"))

#3. 컴파일 훈련
model.compile(loss = "categorical_crossentropy", optimizer = "adam", metrics=["acc"])

es = EarlyStopping(
    monitor = 'val_loss', mode = "min", 
    patience = 5, restore_best_weights= True, 
)

batch_size = 8
start_time = time.time()

history = model.fit(x_train,y_train, epochs=30000, batch_size=batch_size, validation_split=0.2, callbacks = [es])

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
    test_loss = loss,
    r2_score = acc,
    train_ration = train_ration
)