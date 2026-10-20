import datetime
import time

import numpy as np
import pandas as pd

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Dropout, Flatten, GlobalAveragePooling2D, MaxPool2D, MaxPooling2D
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from keras.callbacks import EarlyStopping, ModelCheckpoint,ReduceLROnPlateau
from keras.optimizers import Adam
from keras.applications.vgg16 import VGG16


import my_util
from tqdm_callback import TQDMProgress

path = "./_save/cifar10/"
date = datetime.datetime.now().strftime("%m%d_%H%M")
prefix = "k73_"+date
filename = "_{epoch:04d}-{val_loss:.4f}.keras"
filepath = "".join([path, prefix, filename])

#1. 데이터
(x_train,y_train),(x_test,y_test) = cifar10.load_data()

# 스케일링 1(Minmax)
x_train = x_train/255.
x_test = x_test/255.

ohe = OneHotEncoder(sparse_output=False)
y_train = ohe.fit_transform(y_train.reshape(-1,1))
y_test = ohe.transform(y_test.reshape(-1,1))

#2. 모델
vgg16 = VGG16(
    # weights=None,
    include_top=False,
    input_shape=(32, 32, 3),
    # classes=10,
    # pooling="max",
)

# vgg16.trainable = False
# vgg16.summary()

model = Sequential()
model.add(vgg16)
model.add(Flatten())
model.add(Dense(128,activation="relu"))
model.add(Dense(128,activation="relu"))
model.add(Dense(128,activation="relu"))
model.add(Dense(10,activation="softmax"))

model.summary()


#3. 컴파일 훈련
learning_rate5 = 0.01

model.compile(loss = "categorical_crossentropy", optimizer = Adam(learning_rate=learning_rate5), metrics=["acc"])

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
    patience=20, verbose=1, factor=0.1,
)

tqdm_callback = TQDMProgress()

import gc
import tensorflow as tf

# 에폭 종료 또는 학습 완료 후 실행
tf.keras.backend.clear_session()
gc.collect()

batch_size = 512
start_time = time.time()

history = model.fit(x_train,y_train, verbose=0, epochs=2000, batch_size=batch_size, validation_split=0.2, callbacks = [es,mcp,tqdm_callback,rlr])

train_time = time.time() - start_time

tf.keras.backend.clear_session()
gc.collect()

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
    csv_file_path="cifar10.csv"
)

my_util.leaveTop(path=path, prefix=prefix,subfix=".keras",count=5,mode="min")



#acc 0.92 0.7691