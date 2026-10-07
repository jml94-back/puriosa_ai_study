import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, SimpleRNN, Conv1D, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam

#1. 데이터

datasets = np.array([1,2,3,4,5,6,7,8,9,10])
x = np.array([[1,2,3,],
              [2,3,4,],
              [3,4,5,],
              [4,5,6,],
              [5,6,7,],
              [6,7,8,],
              [7,8,9,],
              ]) #
y = np.array([4,5,6,7,8,9,10])

# print(x.shape, y.shape) #(7, 3) (7,)

x = x.reshape(x.shape[0],x.shape[1],1) #(7,3,1)

#2. 모델
model = Sequential()
model.add(Conv1D(16, kernel_size=2, input_shape = (3,1), )) #3차원 -> 2차원 Dense 연결
model.add(Conv1D(16, kernel_size=2,)) #3차원 -> 2차원 Dense 연결
model.add(Flatten())
# model.add(Dense(32,"relu",))
model.add(Dropout(0.1))
# model.add(Dense(64, activation="relu"))
# model.add(Dense(64, activation="relu"))
model.add(Dense(64, activation="relu"))
# model.add(Dense(128, activation="relu"))
model.add(Dropout(0.1))
model.add(Dense(32))
model.add(Dense(1))

# model.summary()
#3. 컴파일 훈련
learning_rate = 0.001

es = EarlyStopping(
    monitor = 'val_loss', mode = "min", 
    patience = 80, restore_best_weights= True, 
)

rlr = ReduceLROnPlateau(
    monitor="val_loss", mode="auto",
    patience=10, verbose=1, factor=0.7,
)
# tqdm_callback = TQDMProgress()


model.compile(loss = "mse", optimizer = Adam(learning_rate=learning_rate))

model.fit(x,y, epochs=100000,verbose=2,batch_size=1, validation_split=0.2, callbacks = [es,rlr])

#4.
results= model.evaluate(x,y)
print("loss:",results)

x_pred = np.array([8,9,10]).reshape(1,3,1)
y_pred = model.predict(x_pred)

print("y_pred:",y_pred)