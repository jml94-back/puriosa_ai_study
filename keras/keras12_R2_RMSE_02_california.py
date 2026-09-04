import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

import time
import my_util

#target r2:0.55, 
#1. 데이터
datasets = fetch_california_housing()

x = datasets.data
y = datasets.target

print(x.shape)

random_num = 451877
x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=random_num, train_size=0.7)

#2.모델
model = Sequential()
model.add(Dense(16, input_dim = 8))
model.add(Dense(64))
model.add(Dense(16))
model.add(Dense(32))
model.add(Dense(64))
model.add(Dense(1))

#3. 컴파일 훈련
model.compile(loss="mse", optimizer="adam")

batch_size=128

start_time = time.time()

history = model.fit(x_train,y_train,epochs = 300, batch_size = batch_size)

train_time = time.time() - start_time

#4. 평가 예측
loss = model.evaluate(x_test,y_test)
print("loss : ",loss)

y_predict = model.predict(x_test)

from sklearn.metrics import r2_score, mean_squared_error

r2 = r2_score(y_test, y_predict)
print("r2 : ",r2)

def RMSE(y_test, y_predict): #rmse 함수 정의
    return np.sqrt(mean_squared_error(y_test,y_predict))

rmse = RMSE(y_test, y_predict)
print(rmse)


my_util.record_model_csv(
    model = model,
    data_shape = x_train.shape,
    random_num = random_num,
    batch_size = batch_size,
    history = history,
    training_time = train_time,
    test_loss = loss,
    r2_score = r2
)
