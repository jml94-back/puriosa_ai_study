# 강제 종료
# taskkill /F /IM python.exe

import itertools
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
import my_util

#1. 데이터
path = "./_data/santander-ct"
train_csv = pd.read_csv(path+"/train.csv", index_col = 0)


x = train_csv.drop(["target"], axis=1)
y= train_csv["target"]

es = EarlyStopping(
    monitor="val_loss", mode="min", patience=20, restore_best_weights=True
)

random_num = 90

def run_train(nodes, activations):
    os.environ["OMP_NUM_THREADS"] = "2"
    os.environ["TF_NUM_INTRAOP_PARALLELISM_THREADS"] = "2"
    os.environ["TF_NUM_INTEROP_PARALLELISM_THREADS"] = "2"

    global x, y, random_num

    x_train,x_test,y_train,y_test = train_test_split(x,y,
                                                     train_size=0.7,
                                                     random_state=random_num,
                                                     stratify=y
                                                    )
    #2. 모델
    model = Sequential([keras.Input(shape=(200,))])
    model.add(Dense(nodes[0], activation=activations[0]))
    model.add(Dense(nodes[1], activation=activations[1]))
    model.add(Dense(nodes[2], activation=activations[2]))
    model.add(Dense(nodes[3], activation=activations[3]))
    model.add(Dense(nodes[4], activation=activations[4]))
    model.add(Dense(nodes[5], activation=activations[5]))
    model.add(Dense(nodes[6], activation=activations[6]))
    model.add(Dense(nodes[7], activation=activations[7]))
    model.add(Dense(1, activation="sigmoid"))

    model.compile(loss="binary_crossentropy",optimizer="adam",metrics=["acc"])
    batch_size = 1024

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
            csv_file_path="santanber_model.csv",
            r2_score=acc
        )

if __name__ == '__main__':
    # 1. 노드 수 및 활성화 함수 후보 정의
    node_candidates = [ 256, 128, 64, 32]
    base_activations = ['relu','relu','relu','relu','relu','relu', 'linear','linear','linear']

    # 2. 8개 층에 대한 각각의 조합 생성
    node_combinations = list(itertools.product(node_candidates, repeat=8))
    act_combinations = list(set(itertools.permutations(base_activations)))

    # 3. (nodes, activations) 전체 매개변수 조합 생성
    param_combinations = list(itertools.product(node_combinations, act_combinations))

    # 동시 실행할 프로세스 수 (CPU 코어 고려 4~6개 추천)
    num_processes = 3

    with mp.Pool(processes=num_processes) as pool:
        try:
            # 두 인자를 받는 run_train(nodes, activations)에 starmap_async 적용
            result = pool.starmap_async(run_train, param_combinations)
            result.get()
        except KeyboardInterrupt:
            print("\n[알림] Ctrl+C 감지: 모든 프로세스를 종료합니다.")
            pool.terminate()
        finally:
            pool.join()