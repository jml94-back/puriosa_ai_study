import datetime
import time

import numpy as np
import pandas as pd

from keras.datasets import fashion_mnist
from keras.models import Model, Sequential
from keras.layers import Conv2D, Dense, Dropout, Flatten, GlobalAveragePooling2D, Input, MaxPool2D, MaxPooling2D
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from keras.callbacks import EarlyStopping, ModelCheckpoint

import my_util

# path = "./_save/fashion/"
# date = datetime.datetime.now().strftime("%m%d_%H%M")
# prefix = "k40_"+date
# filename = "_{epoch:04d}-{val_loss:.4f}.keras"
# filepath = "".join([path, prefix, filename])

#1. 데이터
(x_train,y_train),(x_test,y_test) = fashion_mnist.load_data()

# print(x_train.shape, y_train.shape) # (60000, 28, 28) (60000,)
# print(x_test.shape, y_test.shape)  #(10000, 28, 28) (10000,)

# print(np.unique(y_train, return_counts=True)) #[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# import matplotlib.pyplot as plt

# plt.imshow(x_train[4342],"gray")
# plt.show()

# 스케일링 2(MaxAbs)
x_train = (x_train-127.5)/127.5
x_test = (x_test-127.5)/127.5

x_train = x_train.reshape(-1,28,28,1)
x_test = x_test.reshape(-1,28,28,1)

ohe = OneHotEncoder(sparse_output=False)
y_train = ohe.fit_transform(y_train.reshape(-1,1))
y_test = ohe.transform(y_test.reshape(-1,1))

#2. 모델구성
# model = Sequential()

# model.add(Conv2D(32, (3,3), input_shape=x_test[0].shape))
# model.add(Dropout(0.2))
# model.add(Conv2D(64, (3,3), activation="relu"))
input1 = Input(shape = x_train[0].shape)
l1 = Conv2D(32, (3,3))(input1)
l2 = Dropout(0.2)(l1)
l3 = Conv2D(64, (3,3), activation="relu")(l2)

# model.add(Dropout(0.3))
# model.add(Conv2D(128, (2,2), activation="relu"))
# model.add(Dropout(0.2))
l4 = Dropout(0.3)(l3)
l5 = Conv2D(128, (2,2), activation="relu")(l4)
l6 = Dropout(0.2)(l5)

# model.add(MaxPooling2D())
# model.add(Conv2D(32, (2,2), activation="relu"))
# model.add(MaxPooling2D())
l7 = MaxPooling2D()(l6)
l8 = Conv2D(32, (2,2), activation="relu")(l7)
l9 = MaxPooling2D()(l8)

# model.add(Conv2D(16, (2,2), activation="relu"))
# model.add(Dropout(0.2))
# model.add(Conv2D(16, (2,2), activation="relu"))
l10 = Conv2D(16, (2,2), activation="relu")(l9)
l11 = Dropout(0.2)(l10)
l12 = Conv2D(16, (2,2), activation="relu")(l11)

# model.add(Dropout(0.2))

# model.add(GlobalAveragePooling2D())
# model.add(Dense(units=128, activation="relu"))
l13 = Dropout(0.2)(l12)
l14 = GlobalAveragePooling2D()(l13)
l15 = Dense(units=128, activation="relu")(l14)

# model.add(Dropout(0.3))
# model.add(Dense(units=64, activation="relu"))
# model.add(Dropout(0.2))
l16 = Dropout(0.3)(l15)
l17 = Dense(units=64, activation="relu")(l16)
l18 = Dropout(0.2)(l17)


# model.add(Dense(units=16, activation="relu"))
# model.add(Dense(10, activation="softmax"))
l19 = Dense(units=16, activation="relu")(l18)
output1 = Dense(10, activation="softmax")(l19)

model = Model(inputs=input1, outputs=output1)

#3. 컴파일 훈련
model.compile(loss = "categorical_crossentropy", optimizer = "adam", metrics=["acc"])

es = EarlyStopping(
    monitor = 'val_loss', mode = "min", 
    patience = 50, restore_best_weights= True, 
)

# mcp = ModelCheckpoint(
#     monitor='val_loss', mode='auto', verbose=1,
#     save_best_only=True, filepath=filepath,
# )

batch_size = 256
start_time = time.time()

history = model.fit(x_train,y_train, verbose=2, epochs=2000, batch_size=batch_size, validation_split=0.3, callbacks = [es]) #,mcp])

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
    csv_file_path="fasion.csv"
)


# my_util.leaveTop(path=path, prefix=prefix, subfix=".keras", count=5, mode="min")



#acc 0.92