import numpy as np

x1 = np.array([1,2,3])
print("x1 = ", x1.shape) #(3,)

x2 = np.array([[1,2,3]])
print("x2 = ", x2.shape) #(1, 3)

#x3 = np.array([[1,2],[1,2,4]]) 행렬 갯수가 다르면 실패
x3 = np.array([[1,2],[1,2]])
print("x3 = ", x3.shape) #(2, 2)

x4 = np.array([[[[1,2,3],[4,5,6]],[[7,8,9],[10,11,12]]],[[[13,14,15],[16,17,18]],[[19,20,21],[22,23,24]]]])
print("x4 = ", x4.shape) #(2, 2, 2, 3)

x5 = np.array([[[1,2,3]],[[1,2,3]]])
print("x5 = ", x5.shape) #(2, 1, 3)

x6 = np.array([[[[1]]],[[[2]]]])
print("x6 = ", x6.shape) #(2, 1, 1, 1)