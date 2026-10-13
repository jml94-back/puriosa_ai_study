# 삼전 시작가 맞추기
# lstm conv1d 앙상블

import pandas as pd
import numpy as np

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.models import Sequential, Model,load_model
from keras.layers import Dense, Dropout, Bidirectional, GRU, Conv1D, Flatten, Input, concatenate
from keras.optimizers import Adam

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler, RobustScaler,StandardScaler,MaxAbsScaler

import time
import datetime
import my_util
from tqdm_callback import TQDMProgress

#1. 데이터
path = "./_data/samsung/"
hynix_csv = pd.read_csv(path + "하이닉스261012.csv", index_col = 0, encoding="cp949", thousands=",")
amore_csv = pd.read_csv(path + "아모레261012.csv", index_col = 0,encoding="cp949", thousands=",")
samsung_csv = pd.read_csv(path + "삼성전자261012.csv", index_col = 0,encoding="cp949", thousands=",")

xhy = hynix_csv[["시가","고가","종가","등락률","거래량"]]#[:1135]
xam = amore_csv[["시가","고가","종가","등락률","거래량"]]#[:1128]
y = np.array(samsung_csv[["시가"]])
# print(xhy.shape,xam.shape,y.shape)

scaler = MinMaxScaler()
xhy = scaler.fit_transform(xhy)
xam = scaler.fit_transform(xam)

xhy = my_util.split_x(xhy,12)
xam = my_util.split_x(xam,3)
# print(xhy.shape,xam.shape,y.shape)
# exit()

#2. 모델
model = load_model("./_save/keras70/k70_jml.keras")

#3. fit

#4.평가 예측
y_pred = model.predict([xhy[:1], xam[:1]])
print("tomorrow:", np.round(y_pred[0],-2))
