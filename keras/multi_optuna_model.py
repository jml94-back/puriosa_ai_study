from functools import lru_cache
import time
import sys

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import my_util

import optuna
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

# 0. 데이터 로딩 및 전처리 전용 함수 (1회만 실행됨)[cite: 1]
@lru_cache(maxsize=1)
def load_data():
    random_num = 90

    path = "./_data/bike-sharing-demand"
    train_csv = pd.read_csv(path + "/train.csv")

    # One-hot encoding for 'season'[cite: 1]
    train_seasons = pd.get_dummies(train_csv["season"], dtype=int)
    train_csv = pd.concat([train_csv, train_seasons], axis=1).rename(str, axis="columns") 

    # 'datetime' 컬럼에서 시간 정보 추출[cite: 1]
    dt_series = pd.to_datetime(train_csv["datetime"])
    train_csv["year"] = dt_series.dt.year
    train_csv["hour"] = dt_series.dt.hour

    # 독립변수(X)와 종속변수(y) 분리[cite: 1]
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

# 1. 후보군 정의[cite: 1]
NODE_CANDIDATES = [32, 64, 128, 256]
ACTIVATION_CANDIDATES = [
    'relu', 
    'gelu', 
    'swish', 
    'mish', 
    'elu', 
    'selu', 
    'leaky_relu', 
    'prelu', 
    'linear'
]

# 활성화 함수 문자열을 Keras 레이어 객체로 안전하게 반환하는 함수
def get_activation_layer(act_name):
    if act_name == 'prelu':
        return layers.PReLU()
    elif act_name == 'leaky_relu':
        return layers.LeakyReLU(alpha=0.2)
    else:
        return layers.Activation(act_name)

# 2. Optuna 전용 Keras Pruning Callback 클래스 정의[cite: 1]
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

        # 1) float 변환 및 NaN/Inf 예외 처리 (SQLite DB Commit 에러 방지)
        current_val = float(current_val)
        if np.isnan(current_val) or np.isinf(current_val):
            self.model.stop_training = True
            raise optuna.exceptions.TrialPruned()

        # 2) 현재 Epoch의 검증 손실을 Optuna에 보고[cite: 1]
        self.trial.report(current_val, step=epoch)

        # 3) 성능 부진 시 조기 종료 및 Pruning Exception 발생[cite: 1]
        if self.trial.should_prune():
            self.model.stop_training = True
            raise optuna.exceptions.TrialPruned()

# 3. Objective 함수 정의[cite: 1]
def objective(trial):
    # 1) 하이퍼파라미터 제안[cite: 1]
    nodes = [
        trial.suggest_categorical(f'n_units_l{i}', NODE_CANDIDATES) 
        for i in range(8)
    ]
    activations = [
        trial.suggest_categorical(f'act_l{i}', ACTIVATION_CANDIDATES) 
        for i in range(8)
    ]
    dropouts = [
        trial.suggest_float(f'dropout_l{i}', 0.0, 0.5, step=0.1)
        for i in range(8)
    ]

    # 2) 데이터 준비[cite: 1]
    x_train, x_test, y_train, y_test = load_data()

    # 3) 모델 생성[cite: 1]
    model = models.Sequential([keras.Input(shape=x_train[0].shape)], name=f"Optuna_Trial_{trial.number}")
    
    for i in range(8):
        model.add(layers.Dense(nodes[i]))
        model.add(get_activation_layer(activations[i]))
        if dropouts[i] > 0.0:
            model.add(layers.Dropout(rate=dropouts[i]))

    model.add(layers.Dense(1))

    # 4) 컴파일 및 훈련[cite: 1]
    batch_size = 1024
    epochs = 20
    model.compile(optimizer='adam', loss='mse', metrics=['mse'])

    start_time = time.time()
    
    # Optuna Pruning Callback 적용[cite: 1]
    history = model.fit(
        x_train, y_train,
        validation_data=(x_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[KerasPruningCallback(trial, monitor='val_loss')],
        verbose=0
    )
    
    training_time = time.time() - start_time

    # 5) 테스트/검증 평가[cite: 1]
    test_loss = float(model.evaluate(x_test, y_test, verbose=0)[0])
    y_pred = model.predict(x_test, verbose=0)
    r2 = float(r2_score(y_test, y_pred))

    # 6) CSV 결과 기록[cite: 1]
    try:
        my_util.record_model_csv(
            model=model,
            data_shape=str(x_train.shape),
            batch_size=batch_size,
            history=history,
            training_time=training_time,
            test_loss=test_loss,
            random_num=90,
            r2_score=r2,
            csv_file_path="optuna_search_log.csv"
        )
    except Exception as e:
        print(f"CSV 기록 중 오류 발생: {e}")

    return test_loss

# 4. Study 생성 및 멀티 프로세싱 실행[cite: 1]
if __name__ == "__main__":
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=5)
    storage_name = "sqlite:///optuna_study.db"
    
    study = optuna.create_study(
        study_name="keras_multi_process",
        storage=storage_name,
        direction="minimize",
        pruner=pruner,
        load_if_exists=True
    )

    # 병렬 멀티프로세싱 실행 (n_jobs=-1 은 사용 가능한 전체 CPU 코어 활용)
    study.optimize(objective, n_trials=3000, n_jobs=-1)

    # 5. 결과 확인[cite: 1]
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

    # 백그라운드 Keras 세션 종료 및 가비지 컬렉션 처리
    tf.keras.backend.clear_session()
    sys.exit(0)