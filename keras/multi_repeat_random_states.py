# 강제 종료
# pkill -f multi_repeat_random_states.py

import os

# TensorFlow 로그 수준 설정 (0: 모든 로그, 1: INFO 숨김, 2: INFO/WARNING 숨김, 3: ERROR만 표시)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# oneDNN 알림 끄기
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import multiprocessing as mp

import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time
# import random
import my_util

#1. 데이터
#따릉이 데이터 불러오기
path = "./_data/santander-ct"
train_csv = pd.read_csv(path+"/train.csv", index_col = 0)


x = train_csv.drop(["target"], axis=1)
y= train_csv["target"]

es = EarlyStopping(
    monitor="val_loss", mode="min", patience=30, restore_best_weights=True
)

def run_train(random_num):
    os.environ["OMP_NUM_THREADS"] = "2"
    os.environ["TF_NUM_INTRAOP_PARALLELISM_THREADS"] = "2"
    os.environ["TF_NUM_INTEROP_PARALLELISM_THREADS"] = "2"

    global x, y

    x_train,x_test,y_train,y_test = train_test_split(x,y,
                                                        train_size=0.7,
                                                        random_state=random_num,
                                                        stratify=y
                                                        )
    #2. 모델
    model = Sequential([keras.Input(shape=(200,))])
    model.add(Dense(128))
    model.add(Dense(32))
    model.add(Dense(64, activation="relu"))
    model.add(Dense(128, activation="relu"))
    model.add(Dense(64, activation="relu"))
    model.add(Dense(64))
    model.add(Dense(32))
    model.add(Dense(1, activation="sigmoid"))

    model.compile(loss="binary_crossentropy",optimizer="adam",metrics=["acc"])
    batch_size = 256

    #3. 컴파일 훈련
    for i in range(3):
        start_time = time.time()

        history = model.fit(x_train,y_train, epochs = 500, batch_size= batch_size, verbose=0, callbacks = [es], validation_split=0.2)

        train_time = time.time() - start_time

        #4. 평가 예측
        loss = model.evaluate(x_test, y_test, verbose=0)

        y_pred = model.predict(x_test)
        y_pred = np.round(y_pred)

        acc = accuracy_score(y_test,y_pred)

        my_util.record_model_csv(
            model = model,
            data_shape = x_train.shape,
            random_num = random_num,
            batch_size = batch_size,
            history = history,
            training_time = train_time,
            test_loss = loss,
            csv_file_path="santanber_randnum.csv",
            r2_score=acc
        )

if __name__ == '__main__':
    seeds = range(1000)
    
    # 동시 실행할 프로세스 수 (CPU 코어 고려 4~6개 추천)
    num_processes = 3


    with mp.Pool(processes=num_processes) as pool:
        try:
            result = pool.map_async(run_train, seeds)
            result.get()
        except KeyboardInterrupt:
            print("\n[알림] Ctrl+C 감지: 모든 프로세스를 종료합니다.")
            pool.terminate()
        finally:
            pool.join()