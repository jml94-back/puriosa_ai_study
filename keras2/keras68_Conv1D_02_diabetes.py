import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Conv1D, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow import keras
from keras.optimizers import Adam

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler, MaxAbsScaler

import time
from tqdm_callback import TQDMProgress
import my_util
import datetime

path = "./_save/diabetes/"
date = datetime.datetime.now().strftime("%m%d_%H%M")
prefix = "k68_"+date
filename = "_{epoch:04d}-{val_loss:.4f}.keras"
filepath = "".join([path, prefix, filename])

#1. 데이터
datasets = load_diabetes()

x = datasets.data
y = datasets.target

# print(x.shape,y.shape) #(442, 10) (442,)
# exit()
random_num = 273
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.7,random_state=random_num)

# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
scaler = RobustScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(-1,5,2)
x_test = x_test.reshape(-1,5,2)

#2. 모델
model = Sequential()
model.add(Conv1D(32, kernel_size=2, input_shape = x_train[0].shape, )) #3차원 -> 2차원 Dense 연결
model.add(Conv1D(64, kernel_size=2,))
model.add(Flatten())
model.add(Dense(64))
model.add(Dropout(0.3))
model.add(Dense(32))
model.add(Dropout(0.2))
model.add(Dense(1))

#3. 컴파일 훈련
learning_rate5 = 0.05
model.compile(loss="mse",optimizer=Adam(learning_rate=learning_rate5))
batch_size = 128
start_time = time.time()

es = EarlyStopping(
    monitor = 'val_loss', mode = "min", 
    patience = 50, restore_best_weights= True, 
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

history = model.fit(x_train,y_train, epochs = 5000, verbose=0,
                    batch_size= batch_size, validation_split=0.33,
                    callbacks = [es,mcp,tqdm_callback,rlr],
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
    sub_score=r2,
    csv_file_path="diabetes.csv",
)

my_util.leaveTop(path=path, prefix=prefix, subfix=".keras", count=5, mode="min")