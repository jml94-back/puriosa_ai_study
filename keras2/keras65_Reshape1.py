import time
import datetime

import numpy as np
import pandas as pd

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Dropout,Input, Flatten,Reshape, MaxPooling2D, GlobalAveragePooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam

from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder

from tqdm_callback import TQDMProgress
import my_util

import tensorflow as tf

# # 8.0GB 물리 메모리 중 7GB(7168)로 하드 한계 설정
# gpus = tf.config.list_physical_devices('GPU')
# if gpus:
#     try:
#         tf.config.set_logical_device_configuration(
#             gpus[0],
#             [tf.config.LogicalDeviceConfiguration(memory_limit=7168)] # MB 단위
#         )
#     except RuntimeError as e:
#         print(e)

path = "./_save/mnist/"
date = datetime.datetime.now().strftime("%m%d_%H%M")
prefix = "k63_"+date
filename = "_{epoch:04d}-{val_loss:.4f}.keras"
filepath = "".join([path, prefix, filename])

#1. 데이터
(x_train,y_train),(x_test,y_test) = mnist.load_data()

# x_train = x_train.reshape(-1,28,28,1)
# x_test = x_test.reshape(-1,28,28,1)

# augment_size = 40000

# # 데이터 증폭
# datagen = ImageDataGenerator(
#     #증폭 변환하는 파라미터들
#     # rescale=1./255,
#     horizontal_flip=True,   # 좌우 반전
#     # vertical_flip=True,     # 상하 반전
#     width_shift_range= 0.1, # 평형 이동
#     height_shift_range=0.1, # 수직 이동
#     rotation_range= 20,      # 각도 조절
#     zoom_range=0.1,         #
#     # shear_range=0.7,        # 좌표 하나를 고정하고 다른 좌표들을 이동
#     fill_mode="nearest",
# )

# randidx = np.random.choice(x_train.shape[0],size=augment_size)

# x_augmented = x_train[randidx].copy()
# y_augmented = y_train[randidx].copy()

# x_augmented = next(datagen.flow(
#     x_augmented,
#     y_augmented,
#     batch_size=augment_size,
#     shuffle=False,
# ))[0]

# x_train = np.concatenate((x_train,x_augmented))
# y_train = np.concatenate((y_train,y_augmented))

# 스케일링 1(Minmax)
x_train = x_train/255.
x_test = x_test/255.

augment_size = 140000

# ohe = OneHotEncoder(sparse_output=False)
# y_train = ohe.fit_transform(y_train.reshape(-1,1))
# y_test = ohe.transform(y_test.reshape(-1,1))



#2. 모델 구성
model = Sequential()
model.add(Input(shape=x_test[0].shape,))
model.add(Reshape((28,28,1)))
model.add(Conv2D(64, (3,3),))
model.add(Conv2D(filters=32, kernel_size=(3,3), activation="relu"))
model.add(Dropout(0.2))
model.add(Conv2D(32, (2,2), activation="relu"))
model.add(MaxPooling2D())
model.add(Conv2D(16, (4,4), activation="relu"))
model.add(Dropout(0.2))
model.add(Conv2D(16, (2,2), activation="relu"))
model.add(Dropout(0.2))
model.add(Conv2D(16, (2,2), activation="relu")) #(20,20,16)
model.add(Dense(8))
model.add(GlobalAveragePooling2D()) #(None, 6400)
model.add(Dense(units=128, activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(units=32, activation="relu"))
model.add(Dense(10, activation="softmax"))

#3. 컴파일 훈련
learning_rate5 = 0.01

model.compile(loss = "sparse_categorical_crossentropy", optimizer = Adam(learning_rate=learning_rate5), metrics=["acc"])

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

# import gc
# import tensorflow as tf

# # 에폭 종료 또는 학습 완료 후 실행
# tf.keras.backend.clear_session()
# gc.collect()

batch_size = 2500
start_time = time.time()

history = model.fit(x_train,y_train, verbose=0, epochs=100, batch_size=batch_size, validation_split=0.3, callbacks = [es,mcp,tqdm_callback,rlr])

train_time = time.time() - start_time

# tf.keras.backend.clear_session()
# gc.collect()

#4. 평가 예측
loss = model.evaluate(x_test,y_test)

print(loss)
y_pred = model.predict(x_test)
# print(y_pred.shape)
# print(y_test.shape)
# exit()
y_pred = np.argmax(y_pred,axis=1)
# y_test = np.argmax(y_test,axis=1)
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
    csv_file_path="mnist.csv"
)

my_util.leaveTop(path=path, prefix=prefix, subfix=".keras", count=5, mode="min")
# 0.9938