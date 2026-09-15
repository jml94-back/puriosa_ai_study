from keras.models import Sequential
from keras.layers import Dense, Conv2D

model = Sequential()
model.add(Conv2D(1, (2,2), input_shape=(10,10,4)))
model.add(Conv2D(5,(2,2)))

model.summary()
"""
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d (Conv2D)             (None, 8, 8, 10)          280  28 * 10     
                                                                 
 conv2d_1 (Conv2D)           (None, 7, 7, 5)           205  41 5      
                                                                 
=================================================================
Total params: 485
w : 1 2
    3 4

b : 0
    
x : 0 0 0 0
    0 5 6 0
    0 7 8 0
    0 0 0 0

1*5 + 2*6 + 3*7 + 4*8 + b

1 3 2
3 8 5
2 5 3
"""

