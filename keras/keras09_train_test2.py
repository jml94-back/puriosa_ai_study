import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])

# x_train = np.array([1,2,3,4,5,6,7])
# y_train = np.array([1,2,3,4,5,6,7])

# x_test =  np.array([8,9,10])
# y_test =  np.array([8,9,10])

#[찾아보기]numpy list 슬라이싱 7:3
x_train = x[:int(len(x)*0.7)]
y_train = y[:int(len(y)*0.7)]

x_test = x[int(len(x)*0.7):]
y_test = y[int(len(y)*0.7):]

#2. 모델
model = Sequential()
model.add(Dense(4,input_dim=3))
model.add(Dense(7))
model.add(Dense(4))
model.add(Dense(3))
model.add(Dense(1))

#3. 컴파일 훈련
model.compile(loss="mse",optimizer="adam")
history = model.fit(x_train,y_train, epochs = 500,batch_size=3)

# 4.평가 예측
loss = model.evaluate(x_test,y_test)
print("loss:",loss)

result = model.predict(np.array([11]))
print("result:",result)