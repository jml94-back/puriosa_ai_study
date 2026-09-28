import numpy as np
import pandas as pd

from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow import keras

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler, MaxAbsScaler

import time
import my_util

path_model = "./_save/bike-sharing-demand/"

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

#2. 모델
model = load_model(path_model+"k31_0914_1410_0010-19741.3223.keras")

#3. 컴파일 훈련

#4. 평가 예측
loss = model.evaluate(x_test, y_test)
print(loss) #3014.0657

y_pred = model.predict(x_test)
r2 = r2_score(y_test, y_pred)
print(r2) #0.48296807611104753

# model.save(path + "keras29_3_save_model.keras")
# model.save_weights(path + "keras29_5_save_weights2.weights.h5")