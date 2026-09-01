import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. data
x=np.array([
    [1,2,3,4,5,6,7,8,9,10],
    [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.5, 1.4, 1.3],
    [9,8,7,6,5,4,3,2,1,0]
]).T
y=np.array([1,2,3,4,5,6,7,8,9,10])

print(x.shape)
print(y.shape)

#2.model
model = Sequential()
model.add(Dense(5, input_dim=3))
model.add(Dense(7))
model.add(Dense(3))
model.add(Dense(1))

#compile, fit
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=500,batch_size = 4)

#4.
loss = model.evaluate(x, y)
print("Loss:", loss)

results = model.predict(np.array([[10,1.3,0]]))
print("Predictions:", results)