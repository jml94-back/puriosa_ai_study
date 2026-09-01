from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

# 1. 데이터
# shape 맞춰주기
x = np.array([[1,6],[2,7],[3,8],[4,9],[5,10]])
y = np.array([1,2,3,4,5])

print(x.shape) #(5, 2)
print(y.shape) #(5,)

# 2. 모델 구성
model = Sequential()
model.add(Dense(5, input_shape=(2,)))
model.add(Dense(10))
model.add(Dense(7))
model.add(Dense(1))

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=1000)

# 4. 평가 예측
loss = model.evaluate(x, y)
print("Loss:", loss)

predictions = model.predict(x)
print("Predictions:", predictions)
z= np.array([[8,6],[3,7],[5,8],[9,9],[15,10]]) #90,41,62,103,164
predictions = model.predict(z)
print("Predictions:", predictions)

