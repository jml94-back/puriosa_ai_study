from keras.preprocessing.text import Tokenizer
import pandas as pd
import numpy as np

text = "나는 지금 진짜 진짜 매우 매우 맛있는 김밥을 엄청 마구 마구 마구 마구 먹었다."

token = Tokenizer() #객체(인스턴스) = 클래스 정의체

token.fit_on_texts([text])
print(token.word_index)