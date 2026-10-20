import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten
import tensorflow as tf
import random

random.seed(333)
np.random.seed(333)
tf.random.set_seed(333)

from keras.applications.vgg16 import VGG16

model = VGG16(
    # weights=None,
    # include_top=False,
    # input_shape=(32, 32, 3),
    # classes=10,
    # pooling="max",
)

model.summary()

"""
VGG16 기본 모델
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ input_layer (InputLayer)             │ (None, 224, 224, 3)         │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
...
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ predictions (Dense)                  │ (None, 1000)                │       4,097,000 │
└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
 Total params: 138,357,544 (527.79 MB)
 Trainable params: 138,357,544 (527.79 MB)
"""

##### include_top = False
# 1. FC layer 없어짐
# 2. input_shape를 원하는 데이터로 변경 가능

