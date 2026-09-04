import csv
import datetime
import os
import sys

def record_model_csv(model, data_shape, random_num, batch_size, history, training_time, test_loss, csv_file_path="model_history_log.csv"):
    # 1. 현재날짜시간
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # 2. 실행 파일명 (Jupyter 환경 등에서는 파일명이 다르게 나올 수 있음)
    file_name = os.path.basename(sys.argv[0])
    
    # 3. 모델 이름
    model_name = model.name
    
    # 4. 모델 구조 (동일한 레이어 타입이 연속되면 출력 노드 수/shape만 표기)
    structure_parts = []
    prev_layer_type = None

    # 5. epochs 추출
    epochs = len(history.history['loss'])

    for layer in model.layers:
        # 레이어 이름 클래스명 추출 (예: 'Dense', 'Conv2D', 'Dropout' 등)
        layer_type = layer.__class__.__name__

        # 출력 형태(output_shape)에서 마지막 차원(노드 수) 가져오기
        # 일부 레이어는 output_shape이 list / TensorShape / None / 멀티출력일 수 있어서 안전하게 처리
        output_shape = getattr(layer, 'output_shape', None)
        if isinstance(output_shape, list):
            # 다중 출력을 가질 경우 첫 번째 shape 선택
            output_shape = output_shape[0] if output_shape else None
        elif output_shape is not None:
            try:
                output_shape = tuple(output_shape)
            except TypeError:
                output_shape = None

        if isinstance(output_shape, (tuple, list)):
            dims = [dim for dim in output_shape if dim is not None]
            dim_str = str(dims[-1]) if dims else "?"
        elif hasattr(layer, 'units') and layer.units is not None:
            dim_str = str(layer.units)
        elif hasattr(layer, 'filters') and layer.filters is not None:
            dim_str = str(layer.filters)
        else:
            dim_str = "?"

        # 이전 레이어와 같은 종류(Type)인 경우 -> 노드 수만 추가
        if layer_type == prev_layer_type:
            structure_parts.append(dim_str)
        # 다른 레이어 종류인 경우 -> '레이어타입 노드수' 형태로 추가
        else:
            structure_parts.append(f"{layer_type} {dim_str}")
            prev_layer_type = layer_type

    # " -> " 로 연결
    model_structure = " -> ".join(structure_parts)
    
    # 5. loss 함수 및 optimizer
    # (문자열로 컴파일된 경우와 객체로 컴파일된 경우를 모두 처리)
    loss_func = model.loss if isinstance(model.loss, str) else model.loss.__name__
    optimizer_name = model.optimizer.name if hasattr(model.optimizer, 'name') else type(model.optimizer).__name__
    
    # 6. first loss, last loss
    first_loss = history.history['loss'][0]
    last_loss = history.history['loss'][-1]
    
    # CSV에 기록할 데이터 리스트
    log_data = [
        current_time, file_name, data_shape, random_num, model_name, model_structure, 
        loss_func, optimizer_name, epochs, batch_size, round(training_time, 4), 
        first_loss, last_loss, test_loss
    ]
    
    # 파일이 존재하지 않으면 헤더(Header)를 먼저 작성
    file_exists = os.path.isfile(csv_file_path)
    
    with open(csv_file_path, mode='a', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "time", "pyfile", "data_shape", "random_num", "model", "model_structure", 
                "loss", "optimizer", "epochs", "batch_size", "train_second", 
                "first loss", "last loss","test_loss"
            ])
        writer.writerow(log_data)
    
    print(f"모델 학습 정보가 '{csv_file_path}'에 기록되었습니다.")