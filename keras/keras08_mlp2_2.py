from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#1. 데이터
x= np.array([range(10), range(21,31), range(201,211)]).T
y= np.array(range(1,11))

#2. 모델
model = Sequential()
model.add(Dense(4,input_dim=3))
model.add(Dense(9))
model.add(Dense(6))
model.add(Dense(3))
model.add(Dense(1))


#3. 컴파일 훈련
model.compile(loss="mse",optimizer="adam")
model.fit(x,y, epochs=1000,batch_size=4)

#4. 평가 예측 [10,31,211]
loss = model.evaluate(x,y)
print("loss:",loss)

result = model.predict(np.array([[10,31,211]]))
print(result)