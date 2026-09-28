import numpy as np
from tensorflow.keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, SimpleRNN,GRU
from keras.callbacks import EarlyStopping, ModelCheckpoint,ReduceLROnPlateau
from keras.optimizers import Adam

#1. 데이터
a = np.array(range(1,11))
size = 4        #timestep 크기


def split_x(dataset, size):
    x = []
    for i in range(len(dataset)-size+1):
        subset = dataset[i:(i+size)]
        x.append(subset)
    return np.array(x)

dataset = split_x(a,size)
x = dataset[:, :-1].reshape(7,3,1)
y = dataset[:, -1]

x_pred = np.array([8,9,10]).reshape(1,3,1)

#2. 모델
model = Sequential()
model.add(SimpleRNN(units=128, input_shape = x[0].shape, )) #3차원 -> 2차원 Dense 연결
# model.add(Dense(64,"relu",))
# model.add(Dropout(0.2))
# model.add(Dense(32))
# model.add(Dropout(0.3))
# model.add(Dense(64,activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(16,activation="relu"))
# model.add(Dense(8))
model.add(Dense(1))

#3. 컴파일 훈련
learning_rate = 0.05

es = EarlyStopping(
    monitor = 'val_loss', mode = "min", 
    patience = 300, restore_best_weights= True, 
)

rlr = ReduceLROnPlateau(
    monitor="val_loss", mode="auto",
    patience=25, verbose=1, factor=0.2,
)
# tqdm_callback = TQDMProgress()


model.compile(loss = "mse", optimizer = Adam(learning_rate=learning_rate))

model.fit(x,y, epochs=100000,verbose=2,batch_size=1, validation_split=0.2, callbacks = [es,rlr])

#4.
results= model.evaluate(x,y)
print("loss:",results)

# print(x_pred)
y_pred = model.predict(x_pred)

print("y_pred:",y_pred)#y_pred: [[10.6605]]