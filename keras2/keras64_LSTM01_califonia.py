import numpy as np
from tensorflow.keras.models import Sequential, load_model
from keras.layers import Dense,Dropout,Bidirectional,GRU
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint,ReduceLROnPlateau
from tensorflow import keras
from keras.optimizers import Adam

from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler, MaxAbsScaler

import time
import datetime
from tqdm_callback import TQDMProgress
import my_util

path = "./_save/california/"
date = datetime.datetime.now().strftime("%m%d_%H%M")
prefix = "k64_"+date
filename = "_{epoch:04d}-{val_loss:.4f}.keras"
filepath = "".join([path, prefix, filename])

#1. 데이터
#캘리포니아 집값 정보
datasets = fetch_california_housing(as_frame=True)

x = datasets.data #(20640, 8)
y = datasets.target

random_num = 3096
x_train,x_test,y_train,y_test = train_test_split(x,y,
                                                 train_size=0.7,
                                                 random_state=random_num
                                                 )

# scaler = MinMaxScaler()
scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(-1,4,2)
x_test = x_test.reshape(-1,4,2)

#2. 모델
model = Sequential()
model.add(Bidirectional(GRU(units=32,) , input_shape = x_train.shape[1:],))
model.add(Dense(13))
model.add(Dropout(0.2))
model.add(Dense(38, activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(27, activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(19, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(7, activation="relu"))
model.add(Dropout(0.1))
model.add(Dense(1))

#3. 컴파일 훈련
learning_rate1 = 1
learning_rate2 = 0.001 #default
learning_rate3 = 0.0001
learning_rate4 = 0.005
learning_rate5 = 0.05
learning_rate6 = 0.009

model.compile(loss="mse",optimizer=Adam(learning_rate=learning_rate1))
batch_size = 128

es = EarlyStopping(
    monitor = 'val_loss', mode = "min", 
    patience = 100, verbose= 0,
    restore_best_weights= True, 
)
mcp = ModelCheckpoint(
    monitor='val_loss', mode='auto', verbose=0,
    save_best_only=True, filepath=filepath,
)

rlr = ReduceLROnPlateau(
    monitor="val_loss", mode="auto",
    patience=10, verbose=1, factor=0.1,
)

tqdm_callback = TQDMProgress()

start_time = time.time()

history = model.fit(x_train,y_train, verbose=0, epochs = 1000, batch_size= batch_size, validation_split=0.2,callbacks = [es,mcp,tqdm_callback,rlr],)

train_time = time.time() - start_time

#4. 평가 예측
loss = model.evaluate(x_test, y_test)
print(loss)

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
    sub_score=r2,
    csv_file_path = "california.csv"
)

my_util.leaveTop(path=path, prefix=prefix, subfix=".keras", count=5, mode="min")