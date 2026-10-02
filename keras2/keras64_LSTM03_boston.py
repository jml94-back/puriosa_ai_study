import numpy as np
from tensorflow.keras.models import Sequential
from keras.layers import Dense,Dropout,Bidirectional,GRU
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam

from tensorflow.keras.datasets import boston_housing
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler, RobustScaler,StandardScaler,MaxAbsScaler

import time
import datetime
import my_util
from tqdm_callback import TQDMProgress

path = "./_save/boston/"
date = datetime.datetime.now().strftime("%m%d_%H%M")
prefix = "k64_"+date
filename = "_{epoch:04d}-{val_loss:.4f}.keras"
filepath = "".join([path, prefix, filename])

#1. 데이터
#보스턴 집값 정보
(x_train, y_train), (x_test,y_test) = boston_housing.load_data()

# print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
random_num=0

# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
scaler = RobustScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
# print(x_test.shape)
# exit()

x_train = x_train.reshape(-1,1,13)
x_test = x_test.reshape(-1,1,13)

#2. 모델
model = Sequential()
model.add(Bidirectional(GRU(units=32,) , input_shape = x_train.shape[1:],))
model.add(Dense(64))
model.add(Dropout(0.3))
model.add(Dense(16))
model.add(Dense(32))
model.add(Dropout(0.3))
model.add(Dense(64))
model.add(Dropout(0.3))
model.add(Dense(64))
model.add(Dropout(0.3))
model.add(Dense(32))
model.add(Dropout(0.3))
model.add(Dense(1))

#3. 컴파일 훈련
learning_rate5 = 0.05
model.compile(loss="mse",optimizer=Adam(learning_rate=learning_rate5))
batch_size = 128

es = EarlyStopping(
    monitor = 'val_loss', mode = "min", 
    patience = 20, restore_best_weights= True, 
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

history = model.fit(x_train,y_train, epochs = 5000,verbose=0,
                    batch_size= batch_size, validation_split=0.33,
                    callbacks = [es,mcp,tqdm_callback,rlr],
                    )

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
    csv_file_path="boston.csv",
)

my_util.leaveTop(path=path, prefix=prefix, subfix=".keras", count=5, mode="min")