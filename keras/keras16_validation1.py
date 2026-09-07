import pandas as pd
import numpy as np

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

import time
import my_util

#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])

x_train = np.array([1,2,3,4,5,6])
y_train = np.array([1,2,3,4,5,6])

x_val=np.array([7,8])
y_val=np.array([7,8])

x_test =  np.array([9,10])
y_test =  np.array([9,10])


#2. 모델
model = Sequential()
model.add(Dense(3,input_dim=1))
model.add(Dense(7))
model.add(Dense(5))
model.add(Dense(2))
model.add(Dense(1))

#3. 컴파일 훈련
model.compile(loss = "mse",optimizer="adam")

batch_size = 4
start_time = time.time()

history = model.fit(x_train,y_train, epochs=100, batch_size=batch_size,
                    validation_data = (x_val,y_val)
                    )

train_time = time.time()-start_time

print(train_time)

#4. 평가 예측
loss = model.evaluate(x_test,y_test)
print("loss:",loss)

y_pred = model.predict(x_test)
print("result:",y_pred)
r2 = r2_score(y_test, y_pred)

my_util.record_model_csv(
    model = model,
    data_shape = x_train.shape,
    random_num = -1,
    batch_size = batch_size,
    history = history,
    training_time = train_time,
    test_loss = loss,
    r2_score = r2
)