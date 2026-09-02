import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes

import time
import random
import my_util

#1.데이터
datasets = load_diabetes()

x = datasets.data
y = datasets.target

#2.모델
model = Sequential()
model.add(Dense(4,input_dim=10))
model.add(Dense(7))
model.add(Dense(3))
model.add(Dense(1))

for i in range(20):
    random_num = random.randint(0, 10000)
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.7,random_state=random_num)

    #3.컴파일 훈련
    model.compile(loss="mse", optimizer="adam")
    start_time = time.time()

    batch_size=32
    history = model.fit(x_train, y_train, epochs = 100, batch_size = 32)

    train_time = time.time() - start_time

    #4.평가 예측
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