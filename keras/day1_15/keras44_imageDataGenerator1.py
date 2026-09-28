import numpy as np

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    #증폭 변환하는 파라미터들
    rescale=1./255,
    horizontal_flip=True,   # 수평 반전
    vertical_flip=True,     # 수직 반전
    width_shift_range= 0.1, # 평형 이동
    height_shift_range=0.1, # 수직 이동
    rotation_range= 5,      # 각도 조절
    zoom_range=1.2,         #
    shear_range=0.7,        # 좌표 하나를 고정하고 다른 좌표들을 이동
    fill_mode="nearest",
)

test_datagen = ImageDataGenerator(
    rescale=1./255
)
#테스트 데이터는 변환하지 않는다. 변환하면 오염된 데이터 취급

path_train="./_data/image/brain/train/"
path_test="./_data/image/brain/test/"

xy_train = train_datagen.flow_from_directory(
    directory=path_train,
    target_size=(100,100),
    color_mode="grayscale",
    batch_size=99999,
    class_mode="binary",
    shuffle=True

)

xy_test = test_datagen.flow_from_directory(
    path_test,
    target_size=(100,100),
    color_mode="grayscale",
    batch_size=99999,
    class_mode="binary",
    # shuffle=False
)

# print(len(xy_train)) #16
# print(xy_train.next()) #이터레이터 첫번째

# print(xy_train[0][0]) #<keras.preprocessing.image.DirectoryIterator object at 0x0000024F239E7FA0>
# print(xy_train[0][0])# 첫번째 배치의 x
# print(xy_train[0][1])# 첫번째 배치의 y

print(xy_train[0][0].shape) # (10, 100, 100, 1)

# import matplotlib.pyplot as plt
# plt.imshow(xy_train[0][0][0],"gray")
# plt.show()

print(type(xy_train)) #<class 'keras.preprocessing.image.DirectoryIterator'>
print(type(xy_train[0])) # <class 'tuple'> 수정불가
print(type(xy_train[0][0])) # <class 'numpy.ndarray'>

