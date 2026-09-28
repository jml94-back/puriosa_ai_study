from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#1. 데이터
x= np.array([range(10), range(21,31), range(201,211)]).T
y= np.array([[1,2,3,4,5,6,7,8,9,10],
             [10,9,8,7,6,5,4,3,2,1]]).transpose()

#2. 모델 구성
model = Sequential()
model.add(Dense(4,input_dim=3))
model.add(Dense(8))
model.add(Dense(6))
model.add(Dense(4))
model.add(Dense(2))


#3. 컴파일 훈련
model.compile(loss="mse",optimizer="adam")
model.fit(x,y, epochs=700,batch_size=4)

#4. 평가 예측 [10,31,211]
loss = model.evaluate(x,y)
print("loss:",loss)

result = model.predict(np.array([[10,31,211]]))
print(result)