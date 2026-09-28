import numpy as np
from tensorflow.keras.models import load_model

np_path = "./_data/npy/catdog_npy/"
me = np.load(np_path+"wlqrp_100.npy")
me=me/255.

path = "./_save/catdog/k44_0918_1631_0037-0.3582.keras"
# model.save(path + "keras29_1_save_model.keras")
model = load_model(path)

y_pred = model.predict(me)
if (np.round(y_pred[0],1) == 0.5):
    print("너는 혼종이다") 
elif (y_pred[0] < 0.5):
    print("너는 고양이다")
    print(100-y_pred[0]*100,"%")
else :
    print("너는 개다")
    print(y_pred[0]*100,"%")