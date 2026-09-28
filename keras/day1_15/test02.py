from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np


# 1. 데이터
# 문자 데이터 적용 가능한지 확인 > 에러 
x = np.array([1,2,3,4,5,6])
y = np.array(["1","a","ㄱ","64","nj","펴"])

# 2. 모델 구성
model = Sequential()
model.add(Dense(3,input_dim=1))
model.add(Dense(5))
model.add(Dense(4))
model.add(Dense(1))


# 3. 컴파일, 훈련
model.compile(loss="mse",optimizer="adam")
# 데이터 배치
model.fit(x,y,epochs=1000) # 배치사이즈 설정(=데이터수/배치사이즈 회 반복). default는 32

# 4. 평가 예측
loss = model.evaluate(x,y)
print("loss: ",loss)
result = model.predict(np.array([10,20,30,40,50,60,70,80]))
print(result)