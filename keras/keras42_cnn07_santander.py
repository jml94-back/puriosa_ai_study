# https://www.kaggle.com/competitions/santander-customer-transaction-prediction/data
import pandas as pd
import numpy as np

from tensorflow.keras.models import Sequential
from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler, MaxAbsScaler

import time
import datetime
import my_util

# path = "./_save/santander/"
# date = datetime.datetime.now().strftime("%m%d_%H%M")
# prefix = "k42_"+date
# filename = "_{epoch:04d}-{val_loss:.4f}.keras"
# filepath = "".join([path, prefix, filename])

#1. 데이터
path = "./_data/santander-ct"
train_csv = pd.read_csv(path+"/train.csv", index_col = 0)

test_csv = pd.read_csv(path+"/test.csv", index_col = 0)
submit_csv = pd.read_csv(path+"/sample_submission.csv", index_col = 0)

# print(train_csv)
# print(test_csv.shape)
# print(submit_csv.shape)
# exit()

x = train_csv.drop(["target"], axis=1)
y= train_csv["target"]

# 판다스 데이터를 가져왔을 때, reshape가 안 되면 실행
# y = y.to_numpy() # == np.array(y)
# y = y.reshape(-1,1)

# print(y, y.shape)
# exit()

y = pd.get_dummies(y)

print(np.unique(y, return_counts=True))

rand_num = 90
train_ration = 0.8
x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=train_ration,random_state=rand_num,stratify=y)

# scaler = MinMaxScaler()
scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)#(200000, 200)
x_test = scaler.transform(x_test)

x_train = np.array(x_train).reshape(-1,10,10,2)
x_test = np.array(x_test).reshape(-1,10,10,2)

#2. 모델
model = Sequential()
model.add(Conv2D(32, (2,2),input_shape=x_test[0].shape)) # (26,26,64)
model.add(Conv2D(filters=32, kernel_size=(2,2), activation="relu")) #(24, 24, 32)
model.add(Dropout(0.2))

model.add(Flatten()) #(None, 6400)
model.add(Dense(units=128, activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(units=64, activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(units=128, activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(units=128, activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(units=32, activation="relu"))
model.add(Dense(2, activation="softmax"))

#3. 컴파일 훈련
model.compile(loss = "mse", optimizer = "adam", metrics=["acc"])

es = EarlyStopping(
    monitor = 'val_loss', mode = "min", 
    patience = 20, restore_best_weights= True, 
)

# mcp = ModelCheckpoint(
#     monitor='val_loss', mode='auto', verbose=1,
#     save_best_only=True, filepath=filepath,
# )

batch_size = 1024
start_time = time.time()

history = model.fit(x_train,y_train, epochs=30000, batch_size=batch_size, validation_split=0.2, callbacks = [es,]) # mcp])

train_time = time.time() - start_time

#4. 평가 예측
loss = model.evaluate(x_test,y_test)

y_pred = model.predict(x_test)
y_pred = np.argmax(y_pred,axis=1)
y_test = np.argmax(y_test,axis=1)
acc = accuracy_score(y_test,y_pred)
print(y_pred)
print(acc)

my_util.record_model_csv(
    model = model,
    data_shape = x_train.shape,
    random_num = rand_num,
    batch_size = batch_size,
    history = history,
    training_time = train_time,
    test_loss = loss,
    sub_score = acc,
    train_ration = train_ration,
    csv_file_path="santander.csv",
)

# y_submint = model.predict(test_csv)

# print(y_submint)
# y_submint = np.argmax(y_submint, axis=1)
# print(y_submint)

# submit_csv['target'] = y_submint
# submit_csv.to_csv(path + "/submit/submission_"+datetime.datetime.now().strftime("%m%d_%H%M")+".csv")

# CPU 64.5855
# GPU 11.6229