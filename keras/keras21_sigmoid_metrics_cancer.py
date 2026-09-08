import numpy as np
import pandas as pd

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_breast_cancer #유방암 관련 데이터셋

import time
import my_util

#1.data
datastes = load_breast_cancer()
# print(datastes.DESCR)
# print(datastes.feature_names)

x = datastes.data
y = datastes.target

# print(type(datastes), type(y)) #sklearn.utils._bunch.Bunch, numpy.ndarray
# print(x.shape, y.shape)
# print(y)

# print(np.unique(y)) #sql distinct 역할
# print(np.unique(y, return_counts=True)) #각 라벨 갯수 확인 (array([0, 1]), array([212, 357]))
# print(pd.DataFrame(y).value_counts()) #1    357  /  0    212
# print(pd.Series(y).value_counts()) #1    357  /  0    212

rand_num = 10425
train_ration = 0.8
x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=train_ration,random_state=rand_num, 
                                                    stratify=y
                                                    )

print(pd.DataFrame(y_train).value_counts()) #285, 170
print(pd.DataFrame(y_test).value_counts())  #72, 42


# print(x_train.shape, y_train.shape) #(455, 30) (455,)
# print(x_test.shape, y_test.shape)   #(114, 30) (114,)

#2. 모델
model = Sequential([keras.Input(shape=(30,))])
model.add(Dense(16))
model.add(Dense(32, activation="relu"))
model.add(Dense(128, activation="relu"))
model.add(Dense(64))
model.add(Dense(64, activation="relu"))
model.add(Dense(128, activation="relu"))
model.add(Dense(64, activation="relu"))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(1, activation="sigmoid")) #이진 분류 마지막은 무조건 sigmoid


#3. 컴파일 훈련
model.compile(loss = "binary_crossentropy", optimizer="adam",
            #   metrics=["accuracy"],
              metrics=["acc"],
              ) #이진 분류 binary_crossentropy

es = EarlyStopping(
    monitor = 'val_loss', mode = "min", 
    patience = 30, restore_best_weights= True, 
)

batch_size = 8
start_time = time.time()

history = model.fit(x_train,y_train, epochs=30000, batch_size=batch_size, validation_split=0.2, callbacks = [es])

train_time = time.time() - start_time

#4. 평가 예측

loss = model.evaluate(x_test,y_test)
# print("loss:", round( loss[0], 4)) 
# print("loss:", round( loss[1], 5))

y_pred = model.predict(x_test)
y_pred = np.round(y_pred) #accuracy_score 에서 필요
# print(y_pred[:10])

acc = accuracy_score(y_test,y_pred)
print(acc)

my_util.record_model_csv(
    model = model,
    data_shape = x_train.shape,
    random_num = rand_num,
    batch_size = batch_size,
    history = history,
    training_time = train_time,
    test_loss = loss,
    r2_score = acc,
    train_ration = train_ration
)