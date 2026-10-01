# https://www.kaggle.com/datasets/stytch16/jena-climate-2009-2016
import os
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"


import pandas as pd
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, SimpleRNN, GRU, Bidirectional
from keras.callbacks import EarlyStopping, ModelCheckpoint,ReduceLROnPlateau
from keras.optimizers import Adam

from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

import time
import datetime
import my_util
from tqdm_callback import TQDMProgress

data_path = "./_data/kaggle_jena"
submit_csv = pd.read_csv(data_path+"/sampleSubmission.csv", index_col = 0)


path = "./_save/jena/"
date = datetime.datetime.now().strftime("%m%d_%H%M")
prefix = "k52_"+date
filename = "_{epoch:04d}-{val_loss:.4f}.keras"
filepath = "".join([path, prefix, filename])

#1. 데이터
#데이터 로드 (scaler 적용 완료)
np_path = "./_data/npy/jena_npy/"
x = np.load(np_path+"x_winter.npy")
y = np.load(np_path+"y_winter.npy")

# print(x.shape,y.shape) #(420408, 144, 13) (420408, 144)

#데이터 자르기
x_train = x[:-288]
y_train = y[144:-144]

x_submit = x[-288:-144]
y_submit = y[-144:]
# print(x_train.shape,y_train.shape) #(420120, 144, 13) (420120, 144)
# print(x_pred.shape) #(144, 144, 13)
# exit()
#train_test
rand_num = 441
train_ration = 0.7
x_train, x_test, y_train, y_test = train_test_split(x_train,y_train, train_size=train_ration, random_state=rand_num)

#2. 모델 구성
model = Sequential()
model.add(Bidirectional(GRU(units=128, return_sequences=True) , input_shape = x[0].shape,)) #3차원 -> 2차원 Dense 연결
model.add(GRU(units=256,))
model.add(Dropout(0.2))
model.add(Dense(256,"relu"))
# model.add(Dropout(0.2))
# model.add(Dense(128,"relu"))
# model.add(Dense(8,activation="relu"))
# model.add(Dense(16,activation="relu"))
model.add(Dense(256))
model.add(Dropout(0.1))
model.add(Dense(144))
model.summary()
learning_rate = 0.02

es = EarlyStopping(
    monitor = 'val_loss', mode = "min", 
    patience = 100, restore_best_weights= True, 
)

mcp = ModelCheckpoint(
    monitor='val_loss', mode='auto', verbose=0,
    save_best_only=True, filepath=filepath,
)

rlr = ReduceLROnPlateau(
    monitor="val_loss", mode="auto",
    patience=10, verbose=1, factor=0.5,
    min_lr=0.000001
)
tqdm_callback = TQDMProgress()


model.compile(loss = "mse", optimizer = Adam(learning_rate=learning_rate))
batch_size = 2200

start_time = time.time()
history = model.fit(x_train,y_train, epochs=1000,verbose=0,batch_size=batch_size, validation_split=0.2, callbacks = [es,rlr,tqdm_callback,mcp])#
train_time = time.time() - start_time

#4.
loss= model.evaluate(x_test,y_test,batch_size)
print("loss:",loss)
y_pred = model.predict(x_test,batch_size)
print(y_pred)
r2 = r2_score(y_test,y_pred)
# y_pred = model.predict(x_pred)

loss2= model.evaluate(x_submit,y_submit)
print("submit loss:",loss2)
y_submit = model.predict(x_submit)

# print("y_pred:",y_pred)

my_util.record_model_csv(
    model = model,
    data_shape = x_train.shape,
    random_num = rand_num,
    batch_size = batch_size,
    history = history,
    training_time = train_time,
    test_loss = loss,
    sub_score = r2,
    train_ration = train_ration,
    csv_file_path="jena.csv"
)
my_util.leaveTop(path=path, prefix=prefix, subfix=".keras", count=5, mode="min")

submit_csv['wd'] = y_submit
submit_csv.to_csv(data_path + "/submit/submission_"+datetime.datetime.now().strftime("%m%d_%H%M")+".csv")

print(my_util.RMSE(y_test,y_pred))
