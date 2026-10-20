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
model.add(Dense(1, input_dim =1,use_bias=False))
# model.add(Dense(2,))
# model.add(Dense(1))

# model.trainable = False #동결
#3. 컴파일 훈련
model.compile(loss = "mse", optimizer = "adam")

model.fit(x,y,1,5000,)

#평가 예측 
print(model.predict(x))

print(model.weights)
