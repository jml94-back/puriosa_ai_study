# https://dacon.io/competitions/open/235576/overview/description
""" 
하이퍼 파라미터

random_states
train_size
레이어 깊이
노드 갯수
epochs
batch_size
"""

import pandas as pd
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow import keras

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, root_mean_squared_error, mean_squared_error

import time
import my_util


#1. 데이터
path = "./_data/ddarung/"
train_csv = pd.read_csv(path + "/train.csv", index_col = 0)
# 결측치 처리 - 
# train_csv = train_csv.dropna() # 결측치 삭제 
train_csv = train_csv.fillna(train_csv.median())


# x,y 분리
x = train_csv.drop(['count'],axis = 1) # axis 0 행 1 열
y = train_csv['count']

rand_num = 858
x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.70, random_state=rand_num)


###submit data
test_csv = pd.read_csv(path + "/test.csv", index_col = 0)
# 결측치 처리 - 평균값 넣기
test_csv = test_csv.fillna(test_csv.median())

#2. 모델
model = Sequential([keras.Input(shape=(9,))])
model.add(Dense(64))
model.add(Dense(16))
model.add(Dense(32))
model.add(Dense(64))
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(1))

#3. 컴파일 훈련
model.compile(loss="mse",optimizer="adam")

start_time = time.time()
batch_size=8

history = model.fit(x_train, y_train, epochs = 700, batch_size = batch_size)

train_time = time.time() - start_time

#4. 평가 예측
loss = model.evaluate(x_test,y_test)

from sklearn.metrics import r2_score
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)

print(np.sqrt(loss))

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

#submission.csv 만들기. count 컬럼 값 삽입

y_submint = model.predict(test_csv)

submission = pd.read_csv(path + "/submission.csv", index_col = 0)
submission['count'] = y_submint
submission.to_csv(path + "/submit/submission_0904_1312.csv")