from sklearn.preprocessing import StandardScaler,MinMaxScaler
import numpy as np

x = np.array([[10],[20],[30],[40],[50],[60],[70],[80],[90],[100],[110],[120]])
#std 단독
std = StandardScaler()
std.fit(x)
x_std = std.transform(x)
print(x_std)

# minmax 후 std
mm = MinMaxScaler()
mm.fit(x)
x_mm = mm.transform(x)

std = StandardScaler()
std.fit(x_mm)
x_mm_std = std.transform(x_mm)
print(x_mm_std)