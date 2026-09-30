import numpy as np
from tensorflow.keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, SimpleRNN,GRU,Bidirectional
from keras.callbacks import EarlyStopping, ModelCheckpoint,ReduceLROnPlateau
from keras.optimizers import Adam
\

#1. 데이터
x=np.array([[1,2,3,],[2,3,4,],[3,4,5,],[4,5,6,],
            [5,6,7,],[6,7,8,],[7,8,9,],[8,9,10,],
            [9,10,11,],[10,11,12,],
            [20,30,40,],[30,40,50,],[40,50,60]
            ])
y=np.array([4,5,6,7,
            8,9,10,11,
            12,13,
            50,60,70])

x_pred=np.array([50,60,70])#target 80

x = x.reshape(13,3,1)
x_pred = x_pred.reshape(1,3,1)

#2. 모델
model = Sequential()
model.add(Bidirectional(LSTM(units=64), input_shape = x[0].shape, )) #3차원 -> 2차원 Dense 연결
# model.add(Dense(16,"relu",))
# model.add(Dropout(0.1))
model.add(Dense(32))
model.add(Dense(16, activation="relu"))
model.add(Dropout(0.1))
model.add(Dense(8))
model.add(Dense(1))

#3. 컴파일 훈련
learning_rate = 0.05

es = EarlyStopping(
    monitor = 'val_loss', mode = "min", 
    patience = 100, restore_best_weights= True, 
)

rlr = ReduceLROnPlateau(
    monitor="val_loss", mode="auto",
    patience=10, verbose=1, factor=0.2,
)
# tqdm_callback = TQDMProgress()


model.compile(loss = "mse", optimizer = Adam(learning_rate=learning_rate))

model.fit(x,y, epochs=100000,verbose=2,batch_size=1, validation_split=0.2, callbacks = [es,rlr])

#4.
results= model.evaluate(x,y)
print("loss:",results)

print(x_pred)
y_pred = model.predict(x_pred)

print("y_pred:",y_pred)