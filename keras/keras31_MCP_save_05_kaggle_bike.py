# https://www.kaggle.com/competitions/bike-sharing-demand/data
import pandas as pd
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, root_mean_squared_log_error
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler, MaxAbsScaler

import time
import datetime
import my_util

path = "./_save/bike-sharing-demand/"
date = datetime.datetime.now().strftime("%m%d_%H%M")
filename = "_{epoch:04d}-{val_loss:.4f}.keras"
filepath = "".join([path,"k31_",date, filename])

#1. 데이터
path = "./_data/bike-sharing-demand"
train_csv = pd.read_csv(path+"/train.csv")

test_csv = pd.read_csv(path+"/test.csv", index_col = 0)
submit_csv = pd.read_csv(path+"/sampleSubmission.csv", index_col = 0)

# print(train_csv.info())
# print(train_csv.describe())
# print(train_csv.isna().sum()) #결측치 확인

train_seasons = pd.get_dummies(train_csv["season"], dtype=int)
train_csv = pd.concat([train_csv,train_seasons], axis=1).rename(str,axis="columns") 

test_seasons = pd.get_dummies(test_csv["season"], dtype=int)
test_csv = pd.concat([test_csv,test_seasons], axis=1).rename(str,axis="columns") 

train_csv["year"] = pd.to_datetime(train_csv.index).year
train_csv["hour"] = pd.to_datetime(train_csv.index).hour

test_csv["year"] = pd.to_datetime(test_csv.index).year
test_csv["hour"] = pd.to_datetime(test_csv.index).hour

x = train_csv.drop(["datetime","casual","registered","count","season"], axis=1)
y = train_csv["count"]

x_submit = test_csv.drop(["season"], axis=1)

rand_num = 2875438
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=rand_num)

# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
scaler = RobustScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# print(x_train.shape, x_submit.shape)
# exit()
#2. 모델
model = Sequential([keras.Input(shape=x_train[0].shape)])
model.add(Dense(128))
model.add(Dense(32))
model.add(Dense(64, activation="gelu"))
model.add(Dense(128, activation="gelu"))
model.add(Dense(64, activation="relu"))
model.add(Dense(64))
model.add(Dense(128, activation="gelu"))
model.add(Dense(64, activation="relu"))
model.add(Dense(32, activation="gelu"))
model.add(Dense(1))

#3. 컴파일 훈련
model.compile(loss = "mse",optimizer="adam")

batch_size = 128

es = EarlyStopping(
    monitor = 'val_loss', mode = "min", 
    patience = 20, restore_best_weights= True, 
)

mcp = ModelCheckpoint(
    monitor='val_loss', mode='auto', verbose=1,
    save_best_only=True, filepath=filepath,
)

start_time=time.time()

history = model.fit(x_train,y_train, epochs=300, batch_size=batch_size, validation_split=0.2, callbacks = [es,mcp])

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
    r2_score = r2,
    csv_file_path="kaggle_bike.csv",
)

y_submint = model.predict(x_submit)

submit_csv['count'] = y_submint
submit_csv.to_csv(path + "/submit/submission_"+datetime.datetime.now().strftime("%m%d_%H%M")+".csv")

