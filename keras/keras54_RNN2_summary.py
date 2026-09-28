import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, SimpleRNN
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam

# (batch size, time step, feature)

#2. 모델
model = Sequential()
model.add(SimpleRNN(units=2, input_shape = (2,3), )) #3차원 -> 2차원 Dense 연결
model.add(Dense(1))

model.summary()
"""
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 simple_rnn (SimpleRNN)      (None, 3)                 15  unit*(unit+input_dim+1)      
 dense (Dense)               (None, 1)                 4         
=================================================================
Total params: 19
Trainable params: 19
Non-trainable params: 0
"""