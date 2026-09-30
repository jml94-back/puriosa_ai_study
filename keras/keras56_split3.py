import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, SimpleRNN,GRU
from keras.callbacks import EarlyStopping, ModelCheckpoint,ReduceLROnPlateau
from keras.optimizers import Adam

from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split

import my_util
# from tqdm_callback import TQDMProgress

a = np.array(range(1,101))
x_pred = np.array(range(96,106)) # 101~106

size =6
dataset = my_util.split_x(a,size)
x = dataset[:, :-1].reshape(-1,5,1)
y = dataset[:, -1]

# print(x.shape) #(95, 5, 1)
# print(y.shape) #(95,)
# exit()

x_pred = my_util.split_x(x_pred,size-1).reshape(-1,5,1)

# print(x_pred) #(6, 5, 1)
# print(x_pred.shape) #(6, 5, 1)
# exit()

rand_num = 90
train_ration = 0.9
x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=train_ration, random_state=rand_num)

#2. 모델
model = Sequential()
model.add(SimpleRNN(units=64, input_shape = x[0].shape, )) #3차원 -> 2차원 Dense 연결
model.add(Dense(16,"relu",))
# model.add(Dropout(0.2))
# model.add(Dense(128,"relu"))
# model.add(Dropout(0.3))
# model.add(Dense(8,activation="relu"))
model.add(Dense(8))
# model.add(Dropout(0.1))
# model.add(Dense(16,activation="relu"))
model.add(Dense(1))

#3. 컴파일 훈련
learning_rate = 0.025

es = EarlyStopping(
    monitor = 'val_loss', mode = "min", 
    patience = 500, restore_best_weights= True, 
)

rlr = ReduceLROnPlateau(
    monitor="val_loss", mode="auto",
    patience=20, verbose=1, factor=0.2,
)
# tqdm_callback = TQDMProgress()


model.compile(loss = "mse", optimizer = Adam(learning_rate=learning_rate))
batch_size = 16
history = model.fit(x_train,y_train, epochs=100000,verbose=0,batch_size=batch_size, validation_split=0.2, callbacks = [es,rlr])#

#4.
loss= model.evaluate(x_test,y_test)
print("loss:",loss)
y_pred = model.predict(x_test)
r2 = r2_score(y_test,y_pred)
# print(x_pred)
y_pred = model.predict(x_pred)

print("y_pred:",y_pred)#y_pred: [[10.6605]]

#target:loss 0.1
#loss: 3.3219282627105713


my_util.record_model_csv(
    model = model,
    data_shape = x_train.shape,
    random_num = 0,
    batch_size = batch_size,
    history = history,
    training_time = 0,
    test_loss = loss,
    sub_score = r2,
    train_ration = 0,
    # csv_file_path="sp.csv"
)