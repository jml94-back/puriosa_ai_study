import numpy as np

from tensorflow.keras.preprocessing.image import load_img,img_to_array
from keras.preprocessing.image import ImageDataGenerator


path = "c:/study/_data/image/ca7e3686-b888-4de1-afdc-b3185618ac9a.jpg"

img = load_img(path, target_size=(150,150))
# print(img)
# import matplotlib.pyplot as plt
# plt.imshow(img)
# plt.show()

arr = img_to_array(img)
arr = np.expand_dims(arr,axis=0)
# print(arr.shape)


# 데이터 증폭
datagen = ImageDataGenerator(
    #증폭 변환하는 파라미터들
    rescale=1./255,
    horizontal_flip=True,   # 좌우 반전
    # vertical_flip=True,     # 상하 반전
    width_shift_range= 0.1, # 평형 이동
    height_shift_range=0.1, # 수직 이동
    rotation_range= 20,      # 각도 조절
    zoom_range=0.1,         #
    # shear_range=0.7,        # 좌표 하나를 고정하고 다른 좌표들을 이동
    fill_mode="nearest",
)

it = datagen.flow(arr,batch_size=2)

import matplotlib.pyplot as plt
fig,ax = plt.subplots(nrows=1,ncols=20,figsize=(5,5))
for i in range(20):
    batch = next(it).reshape(150,150,3)
    print(batch.shape)
    ax[i].imshow(batch)
    # ax[i].axis("off")
plt.show()


# print("저장 완료")