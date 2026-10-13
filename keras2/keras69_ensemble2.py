import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model
from keras.layers import Dense, Input
from keras.callbacks import EarlyStopping

import time

#1.데이터
x1_datastes=np.array([range(100),range(301,401)]).T
                        #0~99          301~400
x2_datastes=np.array([range(101,201),range(411,511),range(150,250)]).T
                           #101~200        411~510        150 ~249
x3_datastes = np.array([range(100),range(301,401),range(77,177),range(33,133)]).T
y = np.array(range(3001,3101))

x1_train, x1_test, _, _ = train_test_split(x1_datastes,y,train_size=0.8,random_state=231)
x2_train, x2_test, y_train, y_test = train_test_split(x2_datastes,y,train_size=0.8,random_state=231)
x3_train, x3_test, _, _ = train_test_split(x3_datastes,y,train_size=0.8,random_state=231)


#2-1.모델
input1 = Input(shape = (2,))
d1 = Dense(10,activation="relu", name="han1")(input1)
d1 = Dense(20,activation="relu", name="han2")(d1)
d1 = Dense(30,activation="relu", name="han3")(d1)
d1 = Dense(40,activation="relu", name="han4")(d1)
# m1 = Model(inputs = input1, outputs=d1)

# m1.summary()
#2-2.모델
input2 = Input(shape = (3,))
d2 = Dense(50, name="dul_1")(input2)
d2 = Dense(40,name="dul_2")(d2)
d2 = Dense(30,name="dul_3")(d2)
d2 = Dense(20,name="dul_4")(d2)
d2 = Dense(10,name="dul_5")(d2)
# m2 = Model(inputs = input2, outputs=d2)

#2-2.모델
input3 = Input(shape = (4,))
d3 = Dense(50, name="set1")(input3)
d3 = Dense(40,name="set2")(d3)
d3 = Dense(30,name="set3")(d3)
d3 = Dense(20,name="set4")(d3)
d3 = Dense(10,name="set5")(d3)

#2-3. 모델
from keras.layers import concatenate,Concatenate

merge1 = concatenate([d1,d2,d3],name = "mg1")
# merge1 = Concatenate(name = "mg1")([d1,d2])

d3 = Dense(10,name="mg2")(merge1)
d3 = Dense(5,name="mg3")(d3)
last_output = Dense(1,name="last")(d3)

model = Model(inputs=[input1,input2,input3], outputs=last_output)

model.summary()

#3. 컴파일 훈련
model.compile(loss="mse",optimizer="adam")
batch_size = 128

es = EarlyStopping(
    monitor = 'val_loss', mode = "min", 
    patience = 100, verbose= 1,
    restore_best_weights= True, 
)

history = model.fit([x1_train,x2_train,x3_train], y_train, epochs = 1000, batch_size= batch_size, validation_split=0.2,callbacks = [es],)

#4. 평가 예측
result = model.evaluate([x1_test,x2_test,x3_test],y_test)
print(result)

x1_pred = np.array([range(100,106),range(400,406)]).T
x2_pred = np.array([range(200,206),range(510,516),range(249,255)]).T
x3_pred = np.array([range(100,106),range(400,406),range(177,183),range(133,139)]).T

y_pred = model.predict([x1_pred,x2_pred,x3_pred])
print(y_pred)