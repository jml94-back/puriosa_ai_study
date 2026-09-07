# https://www.kaggle.com/competitions/bike-sharing-demand/data
import pandas as pd
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow import keras

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from tensorflow.keras.callbacks import EarlyStopping

import time
import my_util
import random

#1. 데이터
x = np.array([ random.randrange(100, 1000) for s in range(10000)])

y = np.array([ x[i]*x[i] for i in range(10000)])

print(x.shape,y.shape)

rand_num = 8674
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=rand_num)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

#2. 모델
model = Sequential([keras.Input(shape=(1,))])
# model.add(Dense(32, activation="relu"))
# model.add(Dense(64))
# model.add(Dense(128, activation="relu"))
# model.add(Dense(32))
# model.add(Dense(64))
# model.add(Dense(128, activation="relu"))
# model.add(Dense(32, activation="relu"))
model.add(Dense(128))
model.add(Dense(32, activation="relu"))
model.add(Dense(64, activation="relu"))
model.add(Dense(128))
model.add(Dense(64, activation="relu"))
model.add(Dense(64, activation="relu"))
model.add(Dense(32, activation="relu"))
model.add(Dense(1))

#3. 컴파일 훈련
model.compile(loss = "mse",optimizer="adam")

batch_size = 512

es = EarlyStopping(
    monitor = 'val_loss', mode = "min", 
    patience = 20, restore_best_weights= True, 
)
start_time=time.time()

history = model.fit(x_train,y_train, epochs=100000, callbacks = [es], batch_size=batch_size, validation_split=0.2)

train_time = time.time() - start_time

#4. 평가 예측
loss = model.evaluate(x_test,y_test)

y_pred = model.predict(x_test)
r2 = r2_score(y_test, y_pred)
# rmsle = root_mean_squared_log_error(y_true=y_test,y_pred=y_pred)
print("r2:",r2)
# print("rmsle:",rmsle)

my_util.record_model_csv(
    model = model,
    data_shape = x_train.shape,
    random_num = rand_num,
    batch_size = batch_size,
    history = history,
    training_time = train_time,
    test_loss = loss,
    r2_score = r2
)

# y_submint = model.predict(test_csv)

# submit_csv['count'] = y_submint
# submit_csv.to_csv(path + "/submit/submission_0907_2.csv")

import matplotlib.pyplot as plt
plt.figure(figsize=(9,6))
plt.rc('font', family='Malgun Gothic')
plt.scatter(x_test,y_test, color="blue", label = "expect") #loss. y값만 넣었을때, x 자동 시간 순.
plt.scatter(x_test,y_pred, color="red", label = "predict") #loss. y값만 넣었을때, x 자동 시간 순.
# plt.plot(history.history["val_loss"][3:], color="blue", label = "val_loss") #val_loss
plt.legend(loc="upper right") #라벨표시 우상단
plt.title("제곱수 Loss")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.grid() #격자표시 추가
# plt.plot(x, result, color="red")
plt.show()

