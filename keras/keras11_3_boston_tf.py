import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from tensorflow.keras.datasets import boston_housing
from sklearn.model_selection import train_test_split

import time
import my_util

#1. 데이터
(x_train, y_train), (x_test,y_test) = boston_housing.load_data()

# print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
random_num=0
# x_train,x_test,y_train,y_test = train_test_split(np.concatenate((x_train,x_test)),np.concatenate((y_train,y_test)),test_size=0.7,random_state=random_num)

#2. 모델
model = Sequential()
model.add(Dense(48,input_dim=13))
model.add(Dense(382))
model.add(Dense(793))
model.add(Dense(291))
model.add(Dense(72))
model.add(Dense(29))
model.add(Dense(1))


#3. 컴파일 훈련
model.compile(loss="mse",optimizer="adam")
start_time = time.time()

batch_size=32
history = model.fit(x_train, y_train, epochs = 650, batch_size = 32)

train_time = time.time() - start_time


#4. 평가 예측
loss = model.evaluate(x_test,y_test)
print(loss)

my_util.record_model_csv(
    model = model,
    data_shape = x_train.shape,
    random_num = random_num,
    batch_size = batch_size,
    history = history,
    training_time = train_time,
    test_loss = loss
)