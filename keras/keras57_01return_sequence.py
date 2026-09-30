import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, SimpleRNN,GRU

from keras.callbacks import EarlyStopping, ModelCheckpoint,ReduceLROnPlateau
from keras.optimizers import Adam

from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split

import my_util
# from tqdm_callback import TQDMProgress

#1. 데이터
x=np.array([[3,4,5,],[4,5,6,],
            [5,6,7,],[6,7,8,],[7,8,9,],[8,9,10,],
            [9,10,11,],[10,11,12,],
            [20,30,40,],[30,40,50,],[40,50,60],
            [1,2,3,],[2,3,4,],
            ])
y=np.array([6,7,
            8,9,10,11,
            12,13,
            50,60,70,
            4,5])

x_pred=np.array([50,60,70])#target 80

x = x.reshape(13,3,1)
x_pred = x_pred.reshape(1,3,1)

#2. 모델
model = Sequential()
model.add(SimpleRNN(units=16, input_shape=(3,1),return_sequences=True))
model.add(SimpleRNN(64,return_sequences=True))
# model.add(Dropout(0.2))
model.add(SimpleRNN(64,return_sequences=True))
model.add(SimpleRNN(32,return_sequences=True))
# model.add(Dropout(0.2))
model.add(SimpleRNN(64 ))
# model.add(Dropout(0.1))
model.add(Dense(128,"relu"))
# model.add(Dropout(0.2))
model.add(Dense(16,))
model.add(Dense(1))


#3. 컴파일 훈련
learning_rate = 0.001

es = EarlyStopping(
    monitor = 'val_loss', mode = "min", 
    patience = 250, restore_best_weights= True, 
)

rlr = ReduceLROnPlateau(
    monitor="val_loss", mode="auto",
    patience=20, verbose=1, factor=0.5,
)
# tqdm_callback = TQDMProgress()


model.compile(loss = "mse", optimizer = Adam(learning_rate=learning_rate))
batch_size = 16
history = model.fit(x,y, epochs=10000,verbose=2,batch_size=batch_size, validation_split=0.3, callbacks = [es,rlr])#

#4.
loss= model.evaluate(x,y)
print("loss:",loss)
y_pred = model.predict(x)
r2 = r2_score(y,y_pred)
# print(x_pred)
y_pred = model.predict(x_pred)

print("y_pred:",y_pred)#y_pred: [[10.6605]]

#target:loss 0.1
#loss: 3.3219282627105713


my_util.record_model_csv(
    model = model,
    data_shape = x.shape,
    random_num = 0,
    batch_size = batch_size,
    history = history,
    training_time = 0,
    test_loss = loss,
    sub_score = r2,
    train_ration = 0,
    # csv_file_path="sp.csv"
)