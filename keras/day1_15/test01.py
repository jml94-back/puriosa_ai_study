from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np


# 1. 데이터
# 제곱도 잘 학습되는지 확인
x = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
y = np.array([1,4,9,16,25,36,49,64,81,100,121,144,169,196,225,256,289,324,361,400])

# 2. 모델 구성
model = Sequential()
model.add(Dense(3,input_dim=1))
model.add(Dense(5))
model.add(Dense(4))
model.add(Dense(1))


# 3. 컴파일, 훈련
model.compile(loss="mse",optimizer="adam")
# 데이터 배치
model.fit(x,y,epochs=1000,batch_size=5) # 배치사이즈 설정(=데이터수/배치사이즈 회 반복). default는 32

# 4. 평가 예측
loss = model.evaluate(x,y)
print("loss: ",loss)
result = model.predict(np.array([10,20,30,40,50,60,70,80]))
print(result)