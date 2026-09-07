# 데이터셋 다운로드 안 될 때 디버그
# import ssl
# ssl._create_default_https_context = ssl._create_unverified_context

# 데이터셋 info 보고 싶어서 사용
# import pandas
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow import keras

from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import r2_score

import time
import my_util

#1. 데이터
#캘리포니아 집값 정보
datasets = fetch_california_housing(as_frame=True)

# datasets = datasets.frame
# datasets.info()

x = datasets.data
y = datasets.target

print(x.shape)

random_num = 30956
x_train,x_test,y_train,y_test = train_test_split(x,y,
                                                 train_size=0.7,
                                                 random_state=random_num
                                                 )

#2. 모델
model = Sequential([keras.Input(shape=(8,))])
model.add(Dense(64))
model.add(Dense(16))
model.add(Dense(32))
model.add(Dense(64))
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(1))

#3. 컴파일 훈련
model.compile(loss="mse",optimizer="adam")
batch_size = 128
start_time = time.time()

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(
    monitor = 'val_loss', mode = "min", 
    patience = 20, restore_best_weights= True, 
)

history = model.fit(x_train,y_train, epochs = 5000,
                    batch_size= batch_size, validation_split=0.33,
                    callbacks = [es],
                    )

train_time = time.time() - start_time

#4. 평가 예측
loss = model.evaluate(x_test, y_test)
print(loss)

y_pred = model.predict(x_test)
print("result:",y_pred)
r2 = r2_score(y_test, y_pred)

my_util.record_model_csv(
    model = model,
    data_shape = x_train.shape,
    random_num = random_num,
    batch_size = batch_size,
    history = history,
    training_time = train_time,
    test_loss = loss,
    r2_score=r2
)

# print("########################history##########################")
# print(history.history)

import matplotlib.pyplot as plt
plt.figure(figsize=(9,6))
plt.rc('font', family='Malgun Gothic')
plt.plot(history.history["loss"][2:], color="red", label = "loss") #loss. y값만 넣었을때, x 자동 시간 순.
plt.plot(history.history["val_loss"][2:], color="blue", label = "val_loss") #val_loss
plt.legend(loc="upper right") #라벨표시 우상단
plt.title("캘리포니아 Loss")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.grid() #격자표시 추가
# plt.plot(x, result, color="red")
plt.show()