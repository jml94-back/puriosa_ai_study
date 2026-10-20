import datetime
import time

import numpy as np
import pandas as pd
import random

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, GlobalAveragePooling2D, MaxPool2D, MaxPooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint,ReduceLROnPlateau
from keras.optimizers import Adam
from keras.applications import VGG19, DenseNet201, EfficientNetB0, MobileNetV2, ResNet101, ResNet50
import tensorflow as tf

from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder


import my_util
from tqdm_callback import TQDMProgress

random.seed(333)
np.random.seed(333)
tf.random.set_seed(333)

#1. 데이터
(x_train,y_train),(x_test,y_test) = cifar10.load_data()

# 스케일링 1(Minmax)
x_train = x_train/255.
x_test = x_test/255.

ohe = OneHotEncoder(sparse_output=False)
y_train = ohe.fit_transform(y_train.reshape(-1,1))
y_test = ohe.transform(y_test.reshape(-1,1))

#2. 모델
model_list = [VGG19(input_shape=x_train.shape[1:],include_top=False),
              ResNet50(input_shape=x_train.shape[1:],include_top=False),
              ResNet101(input_shape=x_train.shape[1:],include_top=False),
              DenseNet201(input_shape=x_train.shape[1:],include_top=False),
            # InceptionV3(),InceptionResNetV2(),
              MobileNetV2(input_shape=x_train.shape[1:],include_top=False),
              EfficientNetB0(input_shape=x_train.shape[1:],include_top=False),]

for premodel in model_list:
    premodel.trainable = False

    model = Sequential()
    model.add(premodel)
    model.add(GlobalAveragePooling2D())
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

    # path = "./_save/cifar10/"
    # date = datetime.datetime.now().strftime("%m%d_%H%M")
    # prefix = "k76_"+premodel.name+"_"+date
    # filename = "_{epoch:04d}-{val_loss:.4f}.keras"
    # filepath = "".join([path, prefix, filename])

    # mcp = ModelCheckpoint(
    #     monitor='val_loss', mode='auto', verbose=0,
    #     save_best_only=True, filepath=filepath,
    # )

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

    batch_size = 64
    start_time = time.time()

    history = model.fit(x_train,y_train, verbose=0, epochs=2000, batch_size=batch_size, validation_split=0.2, callbacks = [es,tqdm_callback,rlr])

    train_time = time.time() - start_time

    tf.keras.backend.clear_session()
    gc.collect()

    #4. 평가 예측
    loss = model.evaluate(x_test,y_test)

    my_util.record_model_csv(
        model = model,
        data_shape = x_train.shape,
        random_num = 0,
        batch_size = batch_size,
        history = history,
        training_time = train_time,
        test_loss = loss[0],
        sub_score = loss[1],
        train_ration = 0,
        csv_file_path="cifar10.csv"
    )

    print("====================================================================")
    print("모델명",premodel.name, "loss,acc:",loss)
    # my_util.leaveTop(path=path, prefix=prefix,subfix=".keras",count=5,mode="min")



#acc 0.92 0.7691