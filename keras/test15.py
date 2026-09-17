from sklearn.preprocessing import MinMaxScaler

x_train = [
    [1,2,3,4],[2,20,200,30000],[2,16,50,1]
]

mmscaler = MinMaxScaler()
mmscaler.fit(x_train)
x_train = mmscaler.transform(x_train)
# x_test = mmscaler.transform(x_test)

print(x_train)