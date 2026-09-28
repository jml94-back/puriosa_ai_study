import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

import time
import my_util

#1. 데이터
x= np.array(range(1,21))
y= np.array([1,2,4,3,5,7,9,3,8,12,13,8,14,15,16,9,6,17,23,21])

random_num=1
x_train,x_test,y_train,y_test = train_test_split(x,y,
                                                 train_size=0.7,
                                                 random_state=random_num
                                                 )

#2. 모델
model = Sequential()
model.add(Dense(4,input_dim=1))
model.add(Dense(7))
model.add(Dense(4))
model.add(Dense(3))
model.add(Dense(1))

#3. 컴파일 훈련
model.compile(loss="mse", optimizer="adam")

batch_size = 5
start_time = time.time()

history = model.fit(x_train,y_train,epochs=80, batch_size=batch_size)

train_time = time.time()-start_time

#4. 평가 예측
loss = model.evaluate(x_test,y_test)
# print(loss)

result = model.predict(x)
# print(result)

my_util.record_model_csv(
    model = model,
    data_shape = x_train.shape,
    random_num = random_num,
    batch_size = batch_size,
    history = history,
    training_time = train_time,
    test_loss = loss
)

import matplotlib.pyplot as plt
plt.scatter(x,y)
plt.plot(x,result)
plt.show()