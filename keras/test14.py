from sklearn.preprocessing import MaxAbsScaler
import numpy as np

x = np.array([[0],[0],[0],[0],[0],[0],[0],[0],[0],[-100],[-200],[0]])
#std 단독
scaler = MaxAbsScaler()
scaler.fit(x)
x_std = scaler.transform(x)
print(x_std)