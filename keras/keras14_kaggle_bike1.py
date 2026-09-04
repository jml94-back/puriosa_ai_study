# https://www.kaggle.com/competitions/bike-sharing-demand/data
import pandas as pd
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow import keras

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, root_mean_squared_log_error

import time
import my_util

#1. 데이터
path = "./_data/bike-sharing-demand"
train_csv = pd.read_csv(path+"/train.csv")

test_csv = pd.read_csv(path+"/test.csv", index_col = 0)
submit_csv = pd.read_csv(path+"/sampleSubmission.csv", index_col = 0)

# print(train_csv.info())
# print(train_csv.describe())
# print(train_csv.isna().sum()) #결측치 확인

# train_csv["year"] = pd.to_datetime(train_csv.index).year
# train_csv["hour"] = pd.to_datetime(train_csv.index).hour

# test_csv["year"] = pd.to_datetime(test_csv.index).year
# test_csv["hour"] = pd.to_datetime(test_csv.index).hour

x = train_csv.drop(["datetime","casual","registered","count"], axis=1)
y = train_csv["count"]

#### year, time 뽑아내보기

rand_num = 675
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.7,random_state=rand_num)

#2. 모델
model = Sequential([keras.Input(shape=(8,))])
model.add(Dense(32, activation="relu"))
model.add(Dense(64))
model.add(Dense(128, activation="relu"))
model.add(Dense(32))
model.add(Dense(64))
model.add(Dense(128, activation="relu"))
model.add(Dense(32, activation="relu"))
model.add(Dense(1))

#3. 컴파일 훈련
model.compile(loss = "mse",optimizer="adam")

batch_size = 128
start_time=time.time()

history = model.fit(x_train,y_train, epochs=100, batch_size=batch_size)

train_time = time.time() - start_time

#4. 평가 예측
loss = model.evaluate(x_test,y_test)

y_pred = model.predict(x_test)
r2 = r2_score(y_test, y_pred)
rmsle = root_mean_squared_log_error(y_true=y_test,y_pred=y_pred)
print("r2:",r2)
print("rmsle:",rmsle)

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

y_submint = model.predict(test_csv)

submit_csv['count'] = y_submint
submit_csv.to_csv(path + "/submit/submission_0904_1.csv")