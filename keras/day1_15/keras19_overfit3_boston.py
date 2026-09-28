import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from tensorflow.keras.datasets import boston_housing
from sklearn.model_selection import train_test_split

from sklearn.metrics import r2_score

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
history = model.fit(x_train, y_train, epochs = 650, batch_size = 32, validation_split=0.33)

train_time = time.time() - start_time


#4. 평가 예측
loss = model.evaluate(x_test,y_test)

y_pred = model.predict(x_test)
r2 = r2_score(y_test, y_pred)

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

import matplotlib.pyplot as plt
plt.figure(figsize=(9,6))
plt.rc('font', family='Malgun Gothic')
plt.plot(history.history["loss"][3:], color="red", label = "loss") #loss. y값만 넣었을때, x 자동 시간 순.
plt.plot(history.history["val_loss"][3:], color="blue", label = "val_loss") #val_loss
plt.legend(loc="upper right") #라벨표시 우상단
plt.title("보스턴 Loss")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.grid() #격자표시 추가
# plt.plot(x, result, color="red")
plt.show()