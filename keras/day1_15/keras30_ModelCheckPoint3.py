import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow import keras

from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler, MaxAbsScaler

import time
import my_util

path = "./_save/california/"

#1. 데이터
#캘리포니아 집값 정보
datasets = fetch_california_housing(as_frame=True)

# datasets = datasets.frame
# datasets.info()

x = datasets.data
y = datasets.target

random_num = 3096
x_train,x_test,y_train,y_test = train_test_split(x,y,
                                                 train_size=0.7,
                                                 random_state=random_num
                                                 )

# min max scaler
# X - min / Max- min -> 0~1 사이에 수렴
# scaler = MinMaxScaler()
scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델
model = Sequential([keras.Input(shape=x_train[0].shape)])
model.add(Dense(13))
model.add(Dense(38, activation="relu"))
model.add(Dense(27, activation="relu"))
model.add(Dense(19, activation="relu"))
model.add(Dense(7, activation="relu"))
model.add(Dense(1))

#3. 컴파일 훈련
model.compile(loss="mse",optimizer="adam")
batch_size = 128

es = EarlyStopping(
    monitor = 'val_loss', mode = "min", 
    patience = 20, verbose= 1,
    restore_best_weights= True, 
)

# mcp 세이브 파일명 만들기
import datetime
date = datetime.datetime.now()
# print(date)         #2026-09-14 11:41:03.793217
# print(type(date))   #<class 'datetime.datetime'>
date = date.strftime("%m%d_%H%M")
# print(date)         #0914_1147
# print(type(date))   #<class 'str'>

filename = "_{epoch:04d}-{val_loss:.4f}.keras"
filepath = "".join([path,"k30_",date, filename])

mcp = ModelCheckpoint(
    monitor='val_loss', mode='auto', verbose=1,
    save_best_only=True, filepath=filepath,
)


start_time = time.time()

history = model.fit(x_train,y_train, epochs = 1000, batch_size= batch_size, validation_split=0.2,callbacks = [es,mcp],)

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
    r2_score=r2,
    csv_file_path = "california.csv"
)

# model.save(path + "keras29_3_save_model.keras")
# model.save_weights(path + "keras29_5_save_weights2.weights.h5")