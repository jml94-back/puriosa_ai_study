import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import LSTM, Embedding, Dense, Dropout, SimpleRNN
import time
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

#1. 데이터
docs = [
    "너무 재미있다","참 최고에요", "참 잘 만든 영화에요",
    "추천하고 싶은 영화입니다", "한 번 더 보고 싶어요", "글쎄",
    "별로에요", "생각보다 지루해요", "연기가 어색해요",
    "재미없어요","너무 재미없다","참 재밌네요",
    "개똥이 바보","말똥이 잘생겼다","길동이 또 구라친다",
    
    "개똥이 잘생겼다","바보 연기가 별로에요", "글쎄 생각보다 너무 잘생겼다", "또 추천하고 싶은 바보"
]
labels = np.array([1,1,1,
                   1,1,0,
                   0,0,0,
                   0,0,1,
                   0,1,0])

token = Tokenizer()
token.fit_on_texts(docs)
# print(token.word_index) # {'참': 1, '너무': 2, '재미있다': 3, '최고에요': 4, '잘': 5, '만든': 6, '영화에요': 7, '추천하고': 8, '싶은': 9, '영화입니다': 10, '한': 11, '번': 12, '더': 13, '보고': 14, '싶어요': 15, '글쎄': 16, '별로에요': 17, '생각보다': 18, '지루해요': 19, '연기가': 20, '어색해요': 21, '재미없어요': 22, '재미없다': 23, '재밌네요': 24, '개똥이': 25, '바보': 26, '말똥이': 27, '잘생겼다': 28, '길동이': 29, '또': 30, '구라친다': 31}

x=token.texts_to_sequences(docs)
# print(x) #[[2, 3], [1, 4], [1, 5, 6, 7], [8, 9, 10], [11, 12, 13, 14, 15], [16], [17], [18, 19], [20, 21], [22], [2, 23], [1, 24], [25, 26], [27, 28], [29, 30, 31]]

#패딩
from tensorflow.keras.preprocessing.sequence import pad_sequences
padded_x = pad_sequences(x,padding = "pre", #뒤 post
                          maxlen = 5, 
                          truncating="post", #default pre 앞에서부터 잘림
                          )

padded_x = to_categorical(padded_x)[:,:,1:]
# print(padded_x)
print(padded_x.shape,labels.shape)
exit()
# padded_x = padded_x.reshape(15,5,1)
rand_num = 8
train_ration = 0.8
x_train, x_test, y_train, y_test = train_test_split(padded_x[:15],labels,train_size=train_ration,random_state=rand_num, 
                                                    stratify=labels
                                                    )

#2. 모델
model = Sequential()
model.add(SimpleRNN(64,input_shape = padded_x[0].shape, use_bias=False,return_sequences=True))
model.add(SimpleRNN(128,))
# model.add(Dense(32))
model.add(Dense(64,activation="relu"))
model.add(Dense(16,activation="relu"))
model.add(Dropout(0.2))

model.add(Dense(1, activation="sigmoid"))

#3. 컴파일 훈련
model.compile(loss = "binary_crossentropy", optimizer="adam",
            #   metrics=["accuracy"],
              metrics=["acc"],
              ) #이진 분류 binary_crossentropy

history = model.fit(x_train,y_train,verbose=2, epochs=300, batch_size=20, validation_split=0.2, )

y_pred = model.predict(x_test)
y_pred = np.round(y_pred) #accuracy_score 에서 필요
# print(x_test)
print(y_pred)

acc = accuracy_score(y_test,y_pred)
print(acc)


y_pred = model.predict(padded_x[15:])
y_pred = np.round(y_pred) #accuracy_score 에서 필요
# print(x_test)
print(y_pred)
# acc
# 1.0

