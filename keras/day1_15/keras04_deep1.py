from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

# 1. 데이터
x = np.array([1,2,3,4,5])
y = np.array([1,2,4,3,5])

# 2. 모델 구성
# 하이퍼 파라미터
model = Sequential()
model.add(Dense(3,input_dim=1))
# model.add(Dense(6,input_dim=3))
# model.add(Dense(5,input_dim=6))
# model.add(Dense(3,input_dim=5))
model.add(Dense(4,input_dim=3))
model.add(Dense(1,input_dim=4))


# 3. 컴파일, 훈련
model.compile(loss="mse",optimizer="adam")
model.fit(x,y,epochs=1000)

# 4. 평가 예측
result = model.predict(np.array([1,2,3,4,5,6,7,8]))
print(result)