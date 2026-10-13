# 삼전 시작가 맞추기
# lstm conv1d 앙상블

import pandas as pd
import numpy as np

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Bidirectional, GRU, Conv1D, Flatten, Input, concatenate
from keras.optimizers import Adam

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler, RobustScaler,StandardScaler,MaxAbsScaler

import time
import datetime
import my_util
from tqdm_callback import TQDMProgress

savepath = "./_save/samsung/"
date = datetime.datetime.now().strftime("%m%d_%H%M")
prefix = "k70_"+date
filename = "_{epoch:04d}-{val_loss:.4f}.keras"
filepath = "".join([savepath, prefix, filename])

#1. 데이터
path = "./_data/samsung/"
hynix_csv = pd.read_csv(path + "하이닉스261012.csv", index_col = 0, encoding="cp949", thousands=",")
amore_csv = pd.read_csv(path + "아모레261012.csv", index_col = 0,encoding="cp949", thousands=",")
samsung_csv = pd.read_csv(path + "삼성전자261012.csv", index_col = 0,encoding="cp949", thousands=",")
# (1300, 5) (2060, 5) (1120, 1)
# 결측치 처리 - 
# train_csv = train_csv.dropna() # 결측치 삭제 
# train_csv = train_csv.fillna(train_csv.median())

xhy = hynix_csv[["시가","고가","종가","등락률","거래량"]]#[:1135]
xam = amore_csv[["시가","고가","종가","등락률","거래량"]]#[:1128]
y = np.array(samsung_csv[["시가"]])
# print(xhy.shape,xam.shape,y.shape)

scaler = MinMaxScaler()
xhy = scaler.fit_transform(xhy)
xam = scaler.fit_transform(xam)

xhy = my_util.split_x(xhy,12)
xam = my_util.split_x(xam,3)
# print(xhy.shape,xam.shape,y.shape)
# exit()

rand_num = 7657
train_ration = 0.7
xhy_train, xhy_test, xam_train, xam_test, y_train, y_test = train_test_split(xhy[1:112], xam[1:112], y[:111], train_size=train_ration, random_state=rand_num)

#2. 모델

hyInput = Input(shape=xhy.shape[1:])
hyM = GRU(64,return_sequences=True)(hyInput)
hyM = Conv1D(64,2,activation="relu")(hyM)
hyM = Conv1D(64,2,activation="relu")(hyM)
hyM = Bidirectional(GRU(128))(hyM)
hyM = Dense(64,activation="relu")(hyM)
hyM = Dense(32,activation="relu")(hyM)
hyM = Dense(12,activation="relu")(hyM)
# m1 = Model(inputs = hyInput, outputs=hyM)

# m1.summary()
amInput = Input(shape=xam.shape[1:])
amM = Conv1D(32,2,activation="relu")(amInput)
amM = Conv1D(64,2,activation="relu")(amM)
amM = GRU(64)(amM)
amM = Dropout(0.1)(amM)
amM = Dense(4,activation="relu")(amM)
# m1 = Model(inputs = amInput, outputs=amM)

# m1.summary()

merge1 = concatenate([hyM,amM],name = "mg1")
merge1 = Dense(32,activation="relu")(merge1)
merge1 = Dense(128,activation="relu")(merge1)
# merge1 = Dense(32,)(merge1)
merge1 = Dense(16,activation="relu")(merge1)
merge1 = Dropout(0.2)(merge1)
merge1 = Dense(1)(merge1)

model = Model(inputs=[hyInput,amInput,], outputs=merge1)
model.summary()

#3. fit
learning_rate = 0.02

es = EarlyStopping(
    monitor = 'val_loss', mode = "min", 
    patience = 200, restore_best_weights= True, 
)

mcp = ModelCheckpoint(
    monitor='val_loss', mode='auto', verbose=0,
    save_best_only=True, filepath=filepath,
)

rlr = ReduceLROnPlateau(
    monitor="val_loss", mode="auto",
    patience=40, verbose=1, factor=0.5,
    min_lr=0.000001
)
tqdm_callback = TQDMProgress()
model.compile(loss = "mse", optimizer = Adam(learning_rate=learning_rate))
batch_size = 200

start_time = time.time()
# print(type(xhy_train),type(xam_train),type(y_train))
history = model.fit([xhy_train, xam_train],y_train, epochs=3000,verbose=0,batch_size=batch_size, validation_split=0.2, callbacks = [es,rlr,tqdm_callback,mcp])#
# history = model.fit([xhy_train,xam_train], y_train, epochs = 10000, batch_size= batch_size, validation_split=0.2,callbacks = [es],)

train_time = time.time() - start_time

#4.평가 예측
loss = model.evaluate([xhy_test,xam_test,],y_test,batch_size)
print("loss: ",loss)

y_pred = model.predict([xhy_test,xam_test,],batch_size)
# print(y_pred)

r2 = r2_score(y_test,y_pred)
print("r2: ",r2)

y_pred = model.predict([xhy[:1], xam[:1]],batch_size)
print("tomorrow:", np.round(y_pred[0],-2))

my_util.record_model_csv(
    model = model,
    data_shape = xhy.shape,
    random_num = rand_num,
    batch_size = batch_size,
    history = history,
    training_time = train_time,
    test_loss = loss,
    sub_score = r2,
    train_ration = train_ration,
    csv_file_path="jena.csv"
)
my_util.leaveTop(path=savepath, prefix=prefix, subfix=".keras", count=3, mode="min")
# 1,059,725,312.0