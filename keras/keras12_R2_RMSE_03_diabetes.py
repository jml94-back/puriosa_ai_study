import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

import time
import my_util

#1. 데이터
datasets = load_diabetes()

x = datasets.data
y = datasets.target

random_num = 273
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.7,random_state=random_num)

#2. 모델
model = Sequential()
model.add(Dense(16,input_dim=10))
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(1))


#3. 컴파일 훈련
model.compile(loss="mse",optimizer="adam")
start_time = time.time()

batch_size=16
history = model.fit(x_train, y_train, epochs = 350, batch_size = batch_size)

train_time = time.time() - start_time


#4. 평가 예측
loss = model.evaluate(x_test,y_test)
print(loss)

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

#target r2: 0.62