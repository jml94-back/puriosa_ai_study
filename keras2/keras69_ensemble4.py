import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model
from keras.layers import Dense, Input
from keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score

import time
import my_util

#1.데이터
x1_datastes=np.array([range(100),range(301,401)]).T
                        #0~99          301~400
# x2_datastes=np.array([range(101,201),range(411,511),range(150,250)]).T
#                            #101~200        411~510        150 ~249
# x3_datastes = np.array([range(100),range(301,401),range(77,177),range(33,133)]).T
y1 = np.array(range(3001,3101))
y2 = np.array(range(13001,13101))

x1_train, x1_test, y1_train, y1_test, y2_train, y2_test= train_test_split(x1_datastes,y1,y2,train_size=0.8,random_state=231)

#2-1.모델
input1 = Input(shape = (2,))
d1 = Dense(10,activation="relu", name="han1")(input1)
d1 = Dense(20,activation="relu", name="han2")(d1)
d1 = Dense(30,activation="relu", name="han3")(d1)
d1 = Dense(40,activation="relu", name="han4")(d1)
# m1 = Model(inputs = input1, outputs=d1)

# m1.summary()
#2-2.모델
# input2 = Input(shape = (3,))
# d2 = Dense(50, name="dul_1")(input2)
# d2 = Dense(40,name="dul_2")(d2)
# d2 = Dense(30,name="dul_3")(d2)
# d2 = Dense(20,name="dul_4")(d2)
# d2 = Dense(10,name="dul_5")(d2)
# # m2 = Model(inputs = input2, outputs=d2)

# #2-3.모델
# input3 = Input(shape = (4,))
# d3 = Dense(50, name="set1")(input3)
# d3 = Dense(40,name="set2")(d3)
# d3 = Dense(30,name="set3")(d3)
# d3 = Dense(20,name="set4")(d3)
# d3 = Dense(10,name="set5")(d3)

#2-3. 모델
from keras.layers import concatenate,Concatenate

# merge1 = concatenate([d1,d2,d3],name = "mg1")
# # merge1 = Concatenate(name = "mg1")([d1,d2])

# d4 = Dense(10,name="mg2")(merge1)
# d4 = Dense(5,name="mg3")(d4)
# last_output = Dense(1,name="last")(d3)

# input4 = Input(shape = (5,))
d5 = Dense(50, name="net1")(d1)
d5 = Dense(40,name="net2")(d5)
d5 = Dense(30,name="net3")(d5)
d5 = Dense(1,name="net4")(d5)

d6 = Dense(50, name="dasut1")(d1)
d6 = Dense(40,name="dasut2")(d6)
d6 = Dense(30,name="dasut3")(d6)
d6 = Dense(1,name="dasut4")(d6)

model = Model(inputs=[input1], outputs=[d5,d6])

# model.summary()

#3. 컴파일 훈련
model.compile(loss="mse",optimizer="adam")
batch_size = 128

es = EarlyStopping(
    monitor = 'val_loss', mode = "min", 
    patience = 100, verbose= 1,
    restore_best_weights= True, 
)

start = time.time()
history = model.fit([x1_train,], [y1_train, y2_train], epochs = 10000, batch_size= batch_size, validation_split=0.2,callbacks = [es],)
train_time = start - time.time()
#4. 평가 예측
result = model.evaluate([x1_test,],[y1_test,y2_test])
r2 = r2_score(x1_test,[y1_test,y2_test])
print(result)

x1_pred = np.array([range(100,106),range(400,406)]).T
# x2_pred = np.array([range(200,206),range(510,516),range(249,255)]).T
# x3_pred = np.array([range(100,106),range(400,406),range(177,183),range(133,139)]).T

y_pred = model.predict([x1_pred,])
print(y_pred)

my_util.record_model_csv(
    model = model,
    data_shape = x1_train.shape,
    random_num = 0,
    batch_size = batch_size,
    history = history,
    training_time = train_time,
    test_loss = result[0],
    sub_score = r2,
    train_ration = 0,
    # csv_file_path="fashion.csv"
)

# my_util.leaveTop(path=path, prefix=prefix, subfix=".keras", count=5, mode="min")