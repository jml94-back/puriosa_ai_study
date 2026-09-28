import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow import keras

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes, load_digits
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler, MaxAbsScaler

import time
import my_util

path = "./_save/digits/"

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
model = load_model(path+"k31_0914_1425_0002-0.3042.keras")

#3. 컴파일 훈련

#4. 평가 예측
loss = model.evaluate(x_test, y_test)
print(loss) #3014.0657

y_pred = model.predict(x_test)
r2 = r2_score(y_test, y_pred)
print(r2) #0.48296807611104753

# model.save(path + "keras29_3_save_model.keras")
# model.save_weights(path + "keras29_5_save_weights2.weights.h5")