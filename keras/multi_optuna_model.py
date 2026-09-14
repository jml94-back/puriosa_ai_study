from functools import lru_cache
import time

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import my_util

import optuna
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import EarlyStopping

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

# 0. 데이터 로딩 및 전처리 전용 함수 (1회만 실행됨)
@lru_cache(maxsize=1)
def load_data():
    random_num = 90

    path = "./_data/bike-sharing-demand"
    train_csv = pd.read_csv(path + "/train.csv")

    # One-hot encoding for 'season'
    train_seasons = pd.get_dummies(train_csv["season"], dtype=int)
    train_csv = pd.concat([train_csv, train_seasons], axis=1).rename(str, axis="columns") 

    # [주의/수정] index가 아닌 'datetime' 컬럼에서 시간 정보 추출
    dt_series = pd.to_datetime(train_csv["datetime"])
    train_csv["year"] = dt_series.dt.year
    train_csv["hour"] = dt_series.dt.hour

    # 독립변수(X)와 종속변수(y) 분리
    x = train_csv.drop(["datetime", "casual", "registered", "count", "season"], axis=1)
    y = train_csv["count"]

    x_train, x_test, y_train, y_test = train_test_split(
            x, y, train_size=0.7, random_state=random_num
        )

    scaler = StandardScaler()

    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)
    
    print("✓ 데이터 로딩 및 전처리 완료 (1회 실행)")
    return x_train, x_test, y_train, y_test

# 1. 후보군 정의
NODE_CANDIDATES = [32, 64, 128, 256]
ACTIVATION_CANDIDATES = ['gelu', 'relu', 'linear', 'swish']

# 2. Optuna 전용 Keras Pruning Callback 클래스 정의
class KerasPruningCallback(tf.keras.callbacks.Callback):
    def __init__(self, trial, monitor='val_loss'):
        super().__init__()
        self.trial = trial
        self.monitor = monitor

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current_val = logs.get(self.monitor)
        
        if current_val is None:
            return

        # 1) 현재 Epoch의 검증 손실을 Optuna에 보고
        self.trial.report(current_val, step=epoch)

        # 2) 성능이 부진하여 Pruning 조건에 걸리면 학습 중단 및 Exception 발생
        if self.trial.should_prune():
            self.model.stop_training = True
            raise optuna.exceptions.TrialPruned()

# 3. Objective 함수 정의
def objective(trial):
    # 1. 하이퍼파라미터 제안
    nodes = [
        trial.suggest_categorical(f'n_units_l{i}', NODE_CANDIDATES) 
        for i in range(13)
    ]
    activations = [
        trial.suggest_categorical(f'act_l{i}', ACTIVATION_CANDIDATES) 
        for i in range(13)
    ]
    dropouts = [
        trial.suggest_categorical(f'dropout_l{i}', [0.0, 0.1, 0.2, 0.3, 0.5])
        for i in range(13)
    ]


    # 1. 데이터 준비
    x_train, x_test, y_train, y_test = load_data()

    # 2. 모델 생성
    model = models.Sequential([keras.Input(shape=x_train[0].shape)],name=f"Optuna_Trial_{trial.number}")
    for i in range(13):
        model.add(layers.Dense(nodes[i], activation=activations[i]))
        if dropouts[i] > 0.0:
            model.add(layers.Dropout(rate=dropouts[i]))
    model.add(layers.Dense(1))

    #3. 컴파일 훈련
    batch_size = 1024
    epochs = 200
    model.compile(optimizer='adam', loss='mse', metrics=['mse'])

    es = EarlyStopping(
    monitor = 'val_loss', mode = "min", 
    patience = 20, restore_best_weights= True, 
    )

    start_time = time.time()
    
    # Optuna Pruning Callback과 연동 (필요 시 적용)
    history = model.fit(
        x_train, y_train,
        validation_data=(x_test, y_test),
        epochs=epochs,
        callbacks=[KerasPruningCallback(trial, monitor='val_loss'),es],
        batch_size=batch_size,
        verbose=0
    )
    
    training_time = time.time() - start_time

    # 4. 테스트/검증 평가
    test_loss = model.evaluate(x_test, y_test, verbose=0)[0]
    y_pred = model.predict(x_test, verbose=0)
    r2 = r2_score(y_test, y_pred)

    # 6. 작성하신 record_model_csv 함수로 CSV에 기록
    try:
        my_util.record_model_csv(
            model=model,
            data_shape=str(x_train.shape),
            batch_size=batch_size,
            history=history,
            training_time=training_time,
            test_loss=test_loss,
            random_num=90,
            r2_score=r2,                   # 필요시 R2 score 계산 후 전달
            csv_file_path="optuna_search_log.csv"
        )
    except Exception as e:
        print(f"CSV 기록 중 오류 발생: {e}")

    return test_loss

# 4. Study 생성 및 실행
if __name__ == "__main__":
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=5)
    
    # 1. 멀티 프로세싱 간 Trial 상태 공유를 위한 SQLite DB 스토리지 생성
    storage_name = "sqlite:///optuna_study.db"
    
    study = optuna.create_study(
        study_name="keras_multi_process",
        storage=storage_name,
        direction="minimize",
        pruner=pruner,
        load_if_exists=True  # 이미 존재할 경우 이어받기
    )

    # 2. n_jobs 옵션으로 병렬 실행할 프로세스 수 지정 (-1은 사용 가능한 모든 CPU 코어 사용)
    # CPU 코어 수에 맞게 n_jobs를 조정하세요 (예: n_jobs=4)
    study.optimize(objective, n_trials=1000, n_jobs=4)

    # 3. 결과 확인
    pruned_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.COMPLETE])

    print("============================================================")
    print(f"전체 시도 횟수 (Study): {len(study.trials)}")
    print(f"완료된 시도 횟수 (Completed): {len(complete_trials)}")
    print(f"조기 종료된 시도 횟수 (Pruned): {len(pruned_trials)}")
    print("============================================================")
    print("최적의 검증 손실 (Best Val Loss):", study.best_value)
    print("최적의 하이퍼파라미터 조합:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")