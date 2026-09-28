import numpy as np
import my_util
from tensorflow.keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, SimpleRNN,GRU
from keras.callbacks import EarlyStopping, ModelCheckpoint,ReduceLROnPlateau
from keras.optimizers import Adam

a = np.array([range(1,11),
              [9,8,7,6,5,4,3,2,1,0]
              ]).T

# print(a)

# print(my_util.split_x(a,4))

dataset = my_util.split_x(a,5)
x = dataset[:, :-1]
y = dataset[:, -1, 1]
x_pred = dataset[-1,1:].reshape(1,4,2)

#2. 모델
model = Sequential()
model.add(SimpleRNN(units=128, input_shape = x[0].shape, )) #3차원 -> 2차원 Dense 연결
model.add(Dense(32,"relu",))
model.add(Dropout(0.2))
# model.add(Dense(32))
# model.add(Dropout(0.3))
model.add(Dense(16, activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(4))
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

# print(x_pred)
y_pred = model.predict(x_pred)

print("y_pred:",y_pred) #y_pred: [[-1.0400227]]