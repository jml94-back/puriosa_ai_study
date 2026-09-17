import numpy as np
from tensorflow.keras.models import Sequential, load_model
from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow import keras

from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler, MaxAbsScaler

import time
import datetime
import my_util

# path = "./_save/california/"
# date = datetime.datetime.now().strftime("%m%d_%H%M")
# prefix = "k42_"+date
# filename = "_{epoch:04d}-{val_loss:.4f}.keras"
# filepath = "".join([path, prefix, filename])

#1. 데이터
#캘리포니아 집값 정보
datasets = fetch_california_housing(as_frame=True)

x = datasets.data #(20640, 8) (20640,)
y = datasets.target

random_num = 3096
x_train,x_test,y_train,y_test = train_test_split(x,y,
                                                 train_size=0.7,
                                                 random_state=random_num
                                                 )

# scaler = MinMaxScaler()
# scaler = StandardScaler()
scaler = MaxAbsScaler()
# scaler = RobustScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

x_train = np.array(x_train).reshape(-1,4,2,1)
x_test = np.array(x_test).reshape(-1,4,2,1)

#2. 모델
model = Sequential()
model.add(Conv2D(32, (2,1),input_shape=x_test[0].shape)) # (26,26,64)
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
model.add(Dense(1,))

#3. 컴파일 훈련
model.compile(loss="mse",optimizer="adam")
batch_size = 128

es = EarlyStopping(
    monitor = 'val_loss', mode = "min", 
    patience = 20, verbose= 1,
    restore_best_weights= True, 
)
# mcp = ModelCheckpoint(
#     monitor='val_loss', mode='auto', verbose=1,
#     save_best_only=True, filepath=path+"keras30_mcp1.keras",
# )


start_time = time.time()

history = model.fit(x_train,y_train, epochs = 1000, batch_size= batch_size, validation_split=0.2,callbacks = [es,])#mcp],)

train_time = time.time() - start_time

#4. 평가 예측
loss = model.evaluate(x_test, y_test)
print(loss)

y_pred = model.predict(x_test)
print("result:",y_pred)
y_pred = y_pred.reshape(-1,)
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

# CPU 21.6584
# GPU 34.0579