import numpy as np

from tensorflow.keras.preprocessing.image import load_img,img_to_array
from tensorflow.keras.models import Sequential, load_model


path = "c:/study/_data/image/horse-or-human/humans/human02-11.png"

img = load_img(path, target_size=(100,100))
# print(img)
# import matplotlib.pyplot as plt
# plt.imshow(img)
# plt.show()

arr = img_to_array(img)
arr = np.expand_dims(arr,axis=0)
# print(arr.shape)

np_path = "./_data/npy/test_npy/"
np.save(np_path+"hh02-11_100.npy",arr)
print("저장 완료")