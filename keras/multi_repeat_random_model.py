# 강제 종료
# taskkill /F /IM python.exe

import itertools
import os

# TensorFlow 로그 수준 설정
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import multiprocessing as mp
import pandas as pd
import numpy as np
import time
import my_util

# 1. 데이터 로드 함수 (독립 전역 함수)
def load_data():
    path = "./_data/bike-sharing-demand"
    train_csv = pd.read_csv(path + "/train.csv")

    train_seasons = pd.get_dummies(train_csv["season"], dtype=int)
    train_csv = pd.concat([train_csv, train_seasons], axis=1).rename(str, axis="columns") 

    train_csv["year"] = pd.to_datetime(train_csv.index).year
    train_csv["hour"] = pd.to_datetime(train_csv.index).hour

    x = train_csv.drop(["datetime", "casual", "registered", "count", "season"], axis=1)
    y = train_csv["count"]
    return x, y

# 2. 멀티프로세싱에서 실행될 독립 함수
def run_train(nodes, activations, x, y, random_num):
    # 프로세스 내부에서 TensorFlow import (Crash 및 OOM 방지)
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.callbacks import EarlyStopping
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score

    os.environ["OMP_NUM_THREADS"] = "2"
    os.environ["TF_NUM_INTRAOP_PARALLELISM_THREADS"] = "2"
    os.environ["TF_NUM_INTEROP_PARALLELISM_THREADS"] = "2"

    # 회귀 문제이므로 stratify 제거
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, train_size=0.7, random_state=random_num
    )

    es = EarlyStopping(
        monitor="val_loss", mode="min", patience=20, restore_best_weights=True
    )

    # input_shape 수정 (x_train.shape[1] 사용)
    model = Sequential([keras.Input(shape=(x_train.shape[1],))])
    for n, act in zip(nodes, activations):
        model.add(Dense(n, activation=act))
    model.add(Dense(1))

    model.compile(loss="mse", optimizer="adam")
    batch_size = 1024

    for i in range(3):
        start_time = time.time()
        history = model.fit(
            x_train, y_train, 
            epochs=500, 
            batch_size=batch_size, 
            verbose=0, 
            callbacks=[es], 
            validation_split=0.2
        )
        train_time = time.time() - start_time

        loss = model.evaluate(x_test, y_test, verbose=0)
        y_pred = model.predict(x_test)
        r2 = r2_score(y_test, y_pred)

        my_util.record_model_csv(
            model=model,
            data_shape=x_train.shape,
            random_num=random_num,
            batch_size=batch_size,
            history=history,
            training_time=train_time,
            test_loss=loss,
            csv_file_path="kaggle_bike_model.csv",
            r2_score=r2
        )

if __name__ == '__main__':
    x, y = load_data()
    random_num = 90

    node_candidates = [32, 64, 128, 256]
    base_activations = ['gelu','gelu','gelu','relu','relu','relu', 'linear','linear',"swish","swish"]

    node_combinations = itertools.product(node_candidates, repeat=8)
    act_combinations = set(itertools.permutations(base_activations, 8))

    # starmap에 전달할 모든 인자 묶음 생성
    param_combinations = (
        (nodes, acts, x, y, random_num)
        for nodes, acts in itertools.product(node_combinations, act_combinations)
    )

    num_processes = 2

    with mp.Pool(processes=num_processes) as pool:
        try:
            result = pool.starmap_async(run_train, param_combinations)
            result.get()
        except KeyboardInterrupt:
            print("\n[알림] Ctrl+C 감지: 모든 프로세스를 종료합니다.")
            pool.terminate()
        finally:
            pool.join()