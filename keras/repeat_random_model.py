import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow import keras

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

import time
import itertools
import my_util

#1. 데이터
#따릉이 데이터 불러오기
data = pd.read_csv('C:/study/_data/ddarung/train.csv',index_col = 0)
data = data.fillna(data.mean())

random_num = 858
x_train,x_test,y_train,y_test = train_test_split(data.drop(['count'],axis = 1),data['count'],
                                                    train_size=0.7,
                                                    random_state=random_num
                                                    )


node_candidates = [128, 64, 32, 16]
node_combinations = list(itertools.product(node_candidates, repeat=6))

best_loss = float('inf')
best_config = None

for nodes in node_combinations:
    #2. 모델
    model = Sequential()
    model.add(keras.Input(shape=(9,)))
    model.add(Dense(nodes[0]))
    model.add(Dense(nodes[1]))
    model.add(Dense(nodes[2]))
    model.add(Dense(nodes[3]))
    model.add(Dense(nodes[4]))
    model.add(Dense(nodes[5]))
    model.add(Dense(1))

    #3. 컴파일 훈련
    model.compile(loss="mse",optimizer="adam")
    batch_size = 128
    start_time = time.time()

    history = model.fit(x_train,y_train, epochs = 20, batch_size= batch_size, verbose=0)# verbose=0 콘솔 출력 생략

    train_time = time.time() - start_time

    #4. 평가 예측
    loss = model.evaluate(x_test, y_test, verbose=0)


    y_predict = model.predict(x_test, verbose=0)
    r2 = r2_score(y_test, y_predict)

    my_util.record_model_csv(
        model = model,
        data_shape = x_train.shape,
        random_num = random_num,
        batch_size = batch_size,
        history = history,
        training_time = train_time,
        test_loss = loss,
        r2_score=r2,
        csv_file_path="ddrri_random_mean_shape.csv",
    )

    if loss < best_loss:
        best_loss = loss
        best_config = nodes