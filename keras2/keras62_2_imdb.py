import datetime
import time

from keras.datasets import imdb

import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score

from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Embedding
from keras.callbacks import EarlyStopping, ModelCheckpoint,ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

import my_util
from tqdm_callback import TQDMProgress

path = "./_save/imdb/"
date = datetime.datetime.now().strftime("%m%d_%H%M")
prefix = "k62_"+date
filename = "_{epoch:04d}-{val_loss:.4f}.keras"
filepath = "".join([path, prefix, filename])

#1. 데이터
(x_train,y_train),(x_test,y_test)= imdb.load_data(
    num_words=1000, #단어 가짓수, 빈도수가 높은 단어순으로 1000개 선택
    maxlen=400, #문장 당 단어수가 100개까지
)

print(np.unique(y_train)) #0 1
print(np.unique(y_test)) #0 1

# print(max(len(i) for i in x_train)) #99
# print(min(len(i) for i in x_train)) #13
# print(max(len(i) for i in x_test))  #99
# print(min(len(i) for i in x_test))  #2

x_train = pad_sequences(x_train,padding = "pre", #뒤 post
                          maxlen = 400, 
                          truncating="post", #default pre 앞에서부터 잘림,
                          value=0
                          )
x_test = pad_sequences(x_test,padding = "pre", #뒤 post
                          maxlen = 400, 
                          truncating="post", #default pre 앞에서부터 잘림
                          value=0
                          )
# print(x_train.shape)
# print(np.unique(x_train)) #1000
# print(np.unique(x_train)) #88586

#2. 모델
model = Sequential()
model.add(Embedding(input_dim=1000,output_dim=200,))#mask_zero=True,
model.add(LSTM(256, use_bias=False))#,return_sequences=True))
model.add(Dropout(0.1))
# model.add(LSTM(128,))
model.add(Dense(64,activation="gelu"))
model.add(Dense(16,activation="relu"))
# model.add(Dropout(0.2))
model.add(Dense(1, activation="sigmoid"))


#3. 학습
learning_rate3 = 0.5


model.compile(loss = "binary_crossentropy", optimizer = Adam(learning_rate=learning_rate3), metrics=["acc"])

es = EarlyStopping(
    monitor = 'val_loss', mode = "min", 
    patience = 500, restore_best_weights= True, 
)

mcp = ModelCheckpoint(
    monitor='val_loss', mode='auto', verbose=0,
    save_best_only=True, filepath=filepath,
)

rlr = ReduceLROnPlateau(
    monitor="val_loss", mode="auto",
    patience=25, verbose=1, factor=0.5,
)

tqdm_callback = TQDMProgress()

batch_size = 1000
start_time = time.time()

history = model.fit(x_train,y_train, verbose=0, epochs=2000, batch_size=batch_size, validation_split=0.2, callbacks = [es,mcp,tqdm_callback,])

train_time = time.time() - start_time

#4. 평가 예측
loss = model.evaluate(x_test,y_test)
print(loss)

# y_pred = model.predict(x_test)
# y_pred = np.argmax(y_pred,axis=1)
# y_test = np.argmax(y_test,axis=1)
# acc = accuracy_score(y_test,y_pred)
# print(acc)

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
    csv_file_path="imdb.csv"
)

my_util.leaveTop(path=path, prefix=prefix,subfix=".keras",count=3,mode="min")