import numpy as np
from tensorflow.keras.models import load_model

np_path = "./_data/npy/test_npy/"
me = np.load(np_path+"hh02-11_100.npy")
me=me/255.

path = "./_save/manwoman/k44_0921_1449_0033-0.1710.keras"
# model.save(path + "keras29_1_save_model.keras")
model = load_model(path)

y_pred = model.predict(me)
if (np.round(y_pred[0],1) == 0.5):
    print("너는 반반인가") 
elif (y_pred[0] < 0.5):
    print("너는 남자다")
    print(100-y_pred[0]*100,"%")
else :
    print("너는 여자다")
    print(y_pred[0]*100,"%")