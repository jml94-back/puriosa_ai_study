import random
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf

# print(tf.__version__)

# random.seed(333)
# np.random.seed(333)
# tf.random.set_seed(333)

#1. 데이터
x= np.array([1,2,3,4,5])
y= np.array([1,2,3,4,5])

#2. 모델
model = Sequential()
model.add(Dense(3, input_dim =1))
model.add(Dense(2,))
model.add(Dense(1))

model.summary()
print("######################### model.weights ##########################")
print(model.weights)

print("\n######################### model.trainable weights ##########################")
print(model.trainable_weights)

print("\n###################################################")
print(len(model.weights)) #6

print("\n######################### 동결 ##########################")
model.trainable = False #동결
print(len(model.weights)) #6
print(len(model.trainable_weights))

model.summary()

# model.compile(loss = "mse", optimizer = "adam")

# model.fit(x,y,1,2000,)

# print(model.weights)
