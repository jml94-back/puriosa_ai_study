import numpy as np
import time
import my_util
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,7,5,7,8,6,10])

#[검색] 사이킷런 
random_num = 333
x_train, x_test, y_train, y_test = train_test_split(x, y, 
                                                    # train_size=0.7, #디폴트 1 - train_size
                                                    # test_size=0.2, #디폴트 = 0.25
                                                    # shuffle=True, #디폴트 =True
                                                    random_state=random_num
                                                    )

#2. 모델 구성
model = Sequential()
model.add(Dense(5,input_dim=1))
model.add(Dense(7))
model.add(Dense(5))
model.add(Dense(3))
model.add(Dense(1))


#3. 컴파일 훈련
model.compile(loss = "mse", optimizer="adam")

batch_size = 4
start_time=time.time()

history = model.fit(x_train,y_train,epochs=50,batch_size=batch_size)

train_time = time.time()-start_time

print("===============================")

#4. 평가 예측
loss = model.evaluate(x_test,y_test)
print(loss)

result = model.predict(x)
print(result)

my_util.record_model_csv(
    model = model,
    data_shape=x_train.shape,
    random_num = random_num,
    batch_size = batch_size,
    history = history,
    training_time = train_time,
    test_loss = loss
)

#그래프
import matplotlib.pyplot as plt
plt.scatter(x, y) #데이터 점 찍기
plt.plot(x, result, color="red")
plt.show()