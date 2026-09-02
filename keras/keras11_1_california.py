# 데이터셋 다운로드 안 될 때 디버그
# import ssl
# ssl._create_default_https_context = ssl._create_unverified_context

# 데이터셋 info 보고 싶어서 사용
# import pandas
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing

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

random_num = 273
x_train,x_test,y_train,y_test = train_test_split(x,y,
                                                 train_size=0.7,
                                                 random_state=random_num
                                                 )

#2. 모델
model = Sequential()
model.add(Dense(13,input_dim=8))
model.add(Dense(38))
model.add(Dense(27))
model.add(Dense(19))
model.add(Dense(7))
model.add(Dense(1))

#3. 컴파일 훈련
model.compile(loss="mse",optimizer="adam")
batch_size = 128
start_time = time.time()

history = model.fit(x_train,y_train, epochs = 100, batch_size= batch_size)

train_time = time.time() - start_time

#4. 평가 예측
loss = model.evaluate(x_test, y_test)
print(loss)

result = model.predict(x)

my_util.record_model_csv(
    model = model,
    data_shape = x_train.shape,
    random_num = random_num,
    batch_size = batch_size,
    history = history,
    training_time = train_time,
    test_loss = loss
)