import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow import keras

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler, MaxAbsScaler

import time
import my_util

path = "./_save/cancer/"

#1. 데이터
datasets = load_breast_cancer()
# print(datastes.DESCR)
# print(datastes.feature_names)

x = datasets.data
y = datasets.target

# print(type(datastes), type(y)) #sklearn.utils._bunch.Bunch, numpy.ndarray
# print(x.shape, y.shape)
# print(y)

# print(np.unique(y)) #sql distinct 역할
# print(np.unique(y, return_counts=True)) #각 라벨 갯수 확인 (array([0, 1]), array([212, 357]))
# print(pd.DataFrame(y).value_counts()) #1    357  /  0    212
# print(pd.Series(y).value_counts()) #1    357  /  0    212

rand_num = 10425
train_ration = 0.8
x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=train_ration,random_state=rand_num, 
                                                    stratify=y
                                                    )

# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
scaler = RobustScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델
model = load_model(path+"k31_0914_1354_0010-0.0689.keras")

#3. 컴파일 훈련

#4. 평가 예측
loss = model.evaluate(x_test, y_test)
print(loss) #3014.0657

y_pred = model.predict(x_test)
r2 = r2_score(y_test, y_pred)
print(r2) #0.48296807611104753

# model.save(path + "keras29_3_save_model.keras")
# model.save_weights(path + "keras29_5_save_weights2.weights.h5")