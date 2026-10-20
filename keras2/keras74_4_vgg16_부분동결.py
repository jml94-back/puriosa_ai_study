import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.applications import VGG16

model = VGG16(
    include_top=False,
    input_shape=(32,32,3),
)

model.trainable = False
model.layers[1].trainable = True


model.summary()

# print(len(model.weights)) #30
# print(len(model.trainable_weights)) #4

import pandas as pd
pd.set_option("max_colwidth", None)
layers = [
    (type(layer).__name__, layer.name, layer.trainable)
    for layer in model.layers
]

results = pd.DataFrame(
    layers,columns=["Type","Name","Trainable"]
)

print(results)
#          Type     Name  Trainable
# 0  Functional    vgg16      False
# 1     Flatten  flatten       True
# 2       Dense    dense       True
# 3       Dense  dense_1       True

