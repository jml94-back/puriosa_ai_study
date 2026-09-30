# https://www.kaggle.com/datasets/stytch16/jena-climate-2009-2016

import pandas as pd
import numpy as np



import time
import datetime

from sklearn.preprocessing import StandardScaler
import my_util
from tqdm_callback import TQDMProgress

path = "./_data/kaggle_jena"
train_csv = pd.read_csv(path+"/jena_climate_2009_2016.csv")
# print(train_csv.shape)
# print(train_csv.isna().sum())


#1,2,11,12월 추출
train_csv['Date Time'] = pd.to_datetime(train_csv['Date Time'], format='%d.%m.%Y %H:%M:%S')
month = train_csv['Date Time'].dt.month.to_numpy()
winter_months = [11, 12, 1, 2]
winter_rows = np.isin(month, winter_months)

x= train_csv.drop(["Date Time","T (degC)"], axis=1)
print(x.shape)
# x = x[:-144]
# print(x.shape)#(420551, 13)

scaler = StandardScaler()
scaler.fit(x[winter_rows])
x = scaler.transform(x)

y = train_csv["T (degC)"]
# y = y[:-144]
print(y.shape)

window_size = 144
x = my_util.split_x(x, window_size)
y = my_util.split_x(y, window_size)

winter_counts = np.concatenate(([0], np.cumsum(winter_rows, dtype=np.int32)))
valid_windows = (
	winter_counts[window_size:] - winter_counts[:-window_size]
) == window_size
x = x[valid_windows]
y = y[valid_windows]

print(x.shape)
print(y.shape)
x = x.astype(np.float16)
y = y.astype(np.float16)

np_path = "./_data/npy/jena_npy/"
np.save(np_path+"x_winter.npy",x)
np.save(np_path+"y_winter.npy",y)