import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data=pd.read_excel(r"C:\Users\师晨迪\Documents\机器学习\zgpa_train.xlsx")
print(data.head())
price=data.loc[:,'close']
print(price.head())
#归一化
price_norm=price/price.max()
print(price_norm)
fig1=plt.figure()
plt.plot(price)
plt.title('close price')
plt.xlabel('time')
plt.ylabel('price')
plt.show()
fig2=plt.figure()
plt.title('close price_norm')
plt.xlabel('time')
plt.ylabel('price')
plt.plot(price_norm)
plt.show()
#define x and y
def extract_data(data,time_step):
    x=[]
    y=[]
    for i in range(len(data)-time_step):
         x.append([a for a in data[i:i+time_step]])
         y.append(data[i+time_step])
    x=np.array(x)
    x=x.reshape(x.shape[0],x.shape[1],1)
    return x,y
time_step=8
x,y=extract_data(price_norm,time_step)
print(x)
print(y)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,SimpleRNN
model = Sequential()
#add rnn layer
model.add(SimpleRNN(5,input_shape=(time_step,1),activation='relu'   ))
model.add(Dense(1,activation='linear'))
#configure the model
model.compile(optimizer='adam',loss='mean_squared_error')
model.summary()
#train the model
print(x.shape,len(y))
y = np.array(y).reshape(-1, 1)
model.fit( x, y, batch_size=30, epochs=200)
#make prediction on the training data
y_train_predict=model.predict(x)*max(price)
y_train=[i*max(price) for i in y]
print(y_train, y_train_predict)
fig3=plt.figure()
plt.title('close price_norm')
plt.plot(y_train,label='real price')
plt.plot(y_train_predict,label='predict price')
plt.xlabel('time')
plt.ylabel('price')
plt.plot(price_norm)
plt.legend()
plt.show()
data_test=pd.read_excel(r"C:\Users\师晨迪\Documents\机器学习\zgpa_test.xlsx")
price_test=data_test.loc[:,'close']
print(price_test.head())
#extract x_test and y_test
price_test_norm=price_test/price_test.max()
x_test_norm,y_test_norm=extract_data(price_test_norm,time_step)
print(x_test_norm.shape,len(y_test_norm))
y_test_predict=model.predict(x_test_norm)*max(price)
y_test=[i*max(price) for i in y_test_norm]
fig4=plt.figure()
plt.title('close price_norm')
plt.plot(y_train,label='real price_test')
plt.plot(y_test_predict,label='predict price_test')
plt.xlabel('time')
plt.ylabel('price')
plt.plot(price_norm)
plt.legend()
plt.show()
result_y_test=np.array(y_test).reshape(-1,1)
result_y_test_predict=y_test_predict
print(result_y_test.shape,result_y_test_predict.shape)
result=np.concatenate((result_y_test,result_y_test_predict),axis=1)
print(result.shape)
result=pd.DataFrame(result,columns=['real price','predict price'])
result.to_csv(r"C:\Users\师晨迪\Documents\机器学习\zgpa_predict_test.xlsx")