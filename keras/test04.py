from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

# 1. 데이터
# shape 맞춰주기
x = np.array([[1,6],[2,7],[3,8],[4,9],[5,10]])
y = np.array([1,2,3,4,5])

# 2. 모델 구성
model = Sequential()
model.add(Dense(5, input_dim=1))
model.add(Dense(7))
model.add(Dense(3))
model.add(Dense(2))

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(y, x, epochs=500, batch_size=3)

# 4. 평가 예측
loss = model.evaluate(y, x)
print("Loss:", loss)

results = model.predict(np.array([1,2,3,4,5]))
print("Predictions:", results)