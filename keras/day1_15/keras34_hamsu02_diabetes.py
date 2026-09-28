import numpy as np
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow import keras

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler, MaxAbsScaler

import time
import my_util
import datetime

path = "./_save/diabetes/"
date = datetime.datetime.now().strftime("%m%d_%H%M")
filename = "_{epoch:04d}-{val_loss:.4f}.keras"
filepath = "".join([path,"k31_",date, filename])

#1. 데이터
datasets = load_diabetes()

x = datasets.data
y = datasets.target

random_num = 273
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.7,random_state=random_num)

# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
scaler = RobustScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델
# model = Sequential([keras.Input(shape=(10,))])
# model.add(Dense(64))
# model.add(Dropout(0.3))
# model.add(Dense(16))
# model.add(Dropout(0.2))
# model.add(Dense(32))
# model.add(Dense(64))
# model.add(Dropout(0.4))
# model.add(Dense(64))
# model.add(Dropout(0.3))
# model.add(Dense(32))
# model.add(Dense(1))

input1 = Input(shape = x_train[0].shape)
dense1 = Dense(64,)(input1)
drop1= Dropout(0.3)(dense1)
dense2 = Dense(16, activation="relu")(drop1)
drop2= Dropout(0.2)(dense2)
dense3 = Dense(32)(drop2)
dense4 = Dense(64)(dense3)
drop4= Dropout(0.4)(dense4)
dense5 = Dense(64)(drop4)
drop5= Dropout(0.3)(dense5)
dense6 = Dense(32)(drop5)
output1 = Dense(1)(dense6)

model = Model(input1,output1)

#3. 컴파일 훈련
model.compile(loss="mse",optimizer="adam")
batch_size = 128
start_time = time.time()

es = EarlyStopping(
    monitor = 'val_loss', mode = "min", 
    patience = 20, restore_best_weights= True, 
)

mcp = ModelCheckpoint(
    monitor='val_loss', mode='auto', verbose=1,
    save_best_only=True, filepath=filepath,
)

history = model.fit(x_train,y_train, epochs = 5000,
                    batch_size= batch_size, validation_split=0.33,
                    callbacks = [es,mcp],
                    )

train_time = time.time() - start_time

#4. 평가 예측
loss = model.evaluate(x_test, y_test)
# print(loss)

y_pred = model.predict(x_test)
# print("result:",y_pred)
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
    csv_file_path="diabetes.csv",
)