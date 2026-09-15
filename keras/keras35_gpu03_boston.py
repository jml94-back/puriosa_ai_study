import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from tensorflow.keras.datasets import boston_housing
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler, RobustScaler,StandardScaler,MaxAbsScaler

import time
import my_util
import datetime

path = "./_save/boston/"
date = datetime.datetime.now().strftime("%m%d_%H%M")
filename = "_{epoch:04d}-{val_loss:.4f}.keras"
filepath = "".join([path,"k31_",date, filename])

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


#2. 모델
model = Sequential([keras.Input(shape=x_train[0].shape)])
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
model.compile(loss="mse",optimizer="adam")
batch_size = 128

es = EarlyStopping(
    monitor = 'val_loss', mode = "min", 
    patience = 20, restore_best_weights= True, 
)

mcp = ModelCheckpoint(
    monitor='val_loss', mode='auto', verbose=1,
    save_best_only=True, filepath=filepath,
)

start_time = time.time()

history = model.fit(x_train,y_train, epochs = 5000,
                    batch_size= batch_size, validation_split=0.33,
                    callbacks = [es,mcp],
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
    r2_score=r2,
    csv_file_path="boston.csv",
)

# GPU 6.28
# CPU 9.6085