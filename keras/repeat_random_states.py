import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

import time
# import random
import my_util

#1. 데이터
#따릉이 데이터 불러오기
path = "./_data/ddarung/"
train_csv = pd.read_csv(path + "/train.csv", index_col = 0)
train_csv.fillna(0,inplace = True)


# #2. 모델
model = Sequential()
model.add(Dense(64,input_dim=9))
model.add(Dense(64))
model.add(Dense(64))
model.add(Dense(1))

model.compile(loss="mse",optimizer="adam")
batch_size = 256

for i in range(100,100000):
    # random_num = random.randint(0, 10000)
    random_num = i
    x_train,x_test,y_train,y_test = train_test_split(train_csv.drop(['count'],axis = 1),train_csv['count'],
                                                    train_size=0.7,
                                                    random_state=random_num
                                                    )

    #3. 컴파일 훈련
    for i in range(3):
        start_time = time.time()

        history = model.fit(x_train,y_train, epochs = 50, batch_size= batch_size, verbose=0)

        train_time = time.time() - start_time

        #4. 평가 예측
        loss = model.evaluate(x_test, y_test)

        y_predict = model.predict(x_test)
        r2 = r2_score(y_test, y_predict)

        my_util.record_model_csv(
            model = model,
            data_shape = x_train.shape,
            random_num = random_num,
            batch_size = batch_size,
            history = history,
            training_time = train_time,
            test_loss = loss,
            csv_file_path="ddrri_random_num2.csv",
            r2_score=r2
        )