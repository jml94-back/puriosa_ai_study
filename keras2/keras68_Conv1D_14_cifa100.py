import time
import datetime

import numpy as np
import pandas as pd

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.datasets import cifar100
from keras.models import Sequential
from keras.layers import GRU, Bidirectional, Conv1D, Conv2D, Dense, Dropout, Flatten, GlobalAveragePooling1D, GlobalAveragePooling2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint,ReduceLROnPlateau

from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder

import my_util
from tqdm_callback import TQDMProgress
import tensorflow as tf

# 8.0GB 물리 메모리 중 7GB(7168MB)로 하드 한계 설정
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.set_logical_device_configuration(
            gpus[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=7168)] # MB 단위
        )
    except RuntimeError as e:
        print(e)

path = "./_save/cifa100/"
date = datetime.datetime.now().strftime("%m%d_%H%M")
prefix = "k68_"+date
filename = "_{epoch:04d}-{val_loss:.4f}.keras"
filepath = "".join([path, prefix, filename])

#1. 데이터
(x_train,y_train),(x_test,y_test) = cifar100.load_data()

# print(x_train.shape, y_train.shape) # (50000, 32, 32, 3) (50000, 1)
# print(x_test.shape, y_test.shape)  #(10000, 32, 32, 3) (10000, 1)

# print(np.unique(y_train, return_counts=True))   #([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
                                                #     17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
                                                #     34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
                                                #     51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67,
                                                #     68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84,
                                                #     85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99]
# import matplotlib.pyplot as plt
# plt.imshow(x_train[4342])
# plt.show()

datagen = ImageDataGenerator(
    #증폭 변환하는 파라미터들
    # rescale=1./255,
    horizontal_flip=True,   # 좌우 반전
    # vertical_flip=True,     # 상하 반전
    width_shift_range= 0.1, # 평형 이동
    height_shift_range=0.1, # 수직 이동
    rotation_range= 20,      # 각도 조절
    zoom_range=0.1,         #
    # shear_range=0.7,        # 좌표 하나를 고정하고 다른 좌표들을 이동
    fill_mode="nearest",
)

augment_size = 40000

randidx = np.random.choice(x_train.shape[0],size=augment_size)

x_augmented = x_train[randidx].copy()
y_augmented = y_train[randidx].copy()

x_augmented = datagen.flow(
    x_augmented,
    y_augmented,
    batch_size=augment_size,
    shuffle=False,
).next()[0]

# 데이터 변환 완료. 데이터 합병
x_train = np.concatenate((x_train,x_augmented))
y_train = np.concatenate((y_train,y_augmented))

# # # 스케일링 1(Minmax)
# x_train = x_train/255.
# x_test = x_test/255.

# 스케일링 2(MaxAbs)
x_train = (x_train-127.5)/127.5
x_test = (x_test-127.5)/127.5

x_train = x_train.reshape(-1,32* 32,3)
x_test = x_test.reshape(-1,32* 32,3)

ohe = OneHotEncoder(sparse_output=False)
y_train = ohe.fit_transform(y_train.reshape(-1,1))
y_test = ohe.transform(y_test.reshape(-1,1))

print(x_train.shape,y_train.shape)


#2. 모델구성
model = Sequential()
model.add(Bidirectional(GRU(units=16, return_sequences=True) , input_shape = x_train.shape[1:], ))
model.add(Conv1D(32, kernel_size=2,))
model.add(Conv1D(64, kernel_size=2,))

model.add(GlobalAveragePooling1D()) #(None, 6400)
model.add(Dense(units=256, activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(units=256, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(units=128, activation="relu"))
model.add(Dense(100, activation="softmax"))

#3. 컴파일 훈련
learning_rate3 = 0.1


model.compile(loss = "categorical_crossentropy", optimizer = Adam(learning_rate=learning_rate3), metrics=["acc"])

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
    patience=10, verbose=1, factor=0.5,
)

tqdm_callback = TQDMProgress()


batch_size = 1000
start_time = time.time()

import gc
import tensorflow as tf

# 에폭 종료 또는 학습 완료 후 실행
tf.keras.backend.clear_session()
gc.collect()

history = model.fit(x_train,y_train, verbose=0, epochs=2000, batch_size=batch_size, validation_split=0.2, callbacks = [es,mcp,tqdm_callback,])

tf.keras.backend.clear_session()
gc.collect()

train_time = time.time() - start_time


#4. 평가 예측
loss = model.evaluate(x_test,y_test)

y_pred = model.predict(x_test)

y_pred = np.argmax(y_pred,axis=1)
y_test = np.argmax(y_test,axis=1)
acc = accuracy_score(y_test,y_pred)
print(acc)

my_util.record_model_csv(
    model = model,
    data_shape = x_train.shape,
    random_num = 0,
    batch_size = batch_size,
    history = history,
    training_time = train_time,
    test_loss = loss[0],
    sub_score = acc,
    train_ration = 0,
    csv_file_path="cifar100.csv"
)

my_util.leaveTop(path=path, prefix=prefix,subfix=".keras",count=5,mode="min")

#acc 0.48