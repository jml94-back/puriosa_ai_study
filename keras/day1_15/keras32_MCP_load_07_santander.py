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

path_model = "./_save/santander/"

#1. 데이터
path = "./_data/santander-ct"
train_csv = pd.read_csv(path+"/train.csv", index_col = 0)

test_csv = pd.read_csv(path+"/test.csv", index_col = 0)
submit_csv = pd.read_csv(path+"/sample_submission.csv", index_col = 0)

print(train_csv.shape)
print(test_csv.shape)
print(submit_csv.shape)

x = train_csv.drop(["target"], axis=1)
y= train_csv["target"]

y = pd.get_dummies(y)

print(np.unique(y, return_counts=True))

rand_num = 90
train_ration = 0.8
x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=train_ration,random_state=rand_num,stratify=y)

# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
scaler = RobustScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델
model = load_model(path_model+"k31_0914_1334_0120-3717.9507.keras")

#3. 컴파일 훈련

#4. 평가 예측
loss = model.evaluate(x_test, y_test)
print(loss) #3014.0657

y_pred = model.predict(x_test)
r2 = r2_score(y_test, y_pred)
print(r2) #0.48296807611104753

# model.save(path + "keras29_3_save_model.keras")
# model.save_weights(path + "keras29_5_save_weights2.weights.h5")