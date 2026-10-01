from keras.preprocessing.text import Tokenizer
import pandas as pd
import numpy as np

text = "나는 지금 진짜 진짜 매우 매우 맛있는 김밥을 엄청 마구 마구 마구 마구 먹었다."
text2 = "개똥이는 기관사를 좋아힌다. 말똥이는 잘생겼다. 길동이는 마구 마구 더 잘생겼다."
token = Tokenizer() #객체(인스턴스) = 클래스 정의체

token.fit_on_texts([text,text2])
print(token.word_index)
print(token.word_counts)

x1 = np.array(token.texts_to_sequences([text]))
x2 = np.array(token.texts_to_sequences([text2]))
# print(x) #[[5, 6, 2, 2, 3, 3, 7, 8, 9, 1, 1, 1, 1, 10], [11, 12, 13, 14, 4, 15, 1, 1, 16, 4]]

from keras.utils import to_categorical
y1 = to_categorical(x1,17)
y2 = to_categorical(x2,17)
print(y1.shape,y2.shape)


# y = pd.get_dummies(np.array(x).reshape(14,), dtype=int)
# print(y)

# from sklearn.preprocessing import OneHotEncoder
# encoder = OneHotEncoder(sparse_output=False) #sparse_output 끄기
# y2 = encoder.transform(np.array(x[1]).reshape(len(x1),))
# y1 = encoder.transform(np.array(x[0]).reshape(len(x[0]),))

# y = np.concatenate([y1,y2])
# print(y)