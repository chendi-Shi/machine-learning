from statistics import LinearRegression

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data_train=pd.read_excel(r"C:\Users\师晨迪\Documents\机器学习\T-R-train.xlsx")
print(data_train.to_string())
#define x_train and y_train
x_train=data_train.loc[:,'T']
y_train=data_train.loc[:,('rate')]
#visualize the result
#观察欠拟合
fig1=plt.figure()
plt.scatter(x_train,y_train)
plt.title=('raw data')
plt.xlabel=('temperature')
plt.ylabel=('rate')
plt.show()
#linear regression model
x_train=np.array(x_train).reshape(-1,1)
from sklearn.linear_model import LinearRegression
lr1=LinearRegression()
lr1.fit(x_train,y_train)
data_test=pd.read_excel(r"C:\Users\师晨迪\Documents\机器学习\T-R-test.xlsx")
print(data_test.to_string())
x_test=data_test.loc[:,'T']
y_test=data_test.loc[:,('rate')]
x_test=np.array(x_test).reshape(-1,1)
#make prediction on the training and testing data
y_train_pred=lr1.predict(x_train)
y_test_pred=lr1.predict(x_test)
from sklearn.metrics import r2_score
r2_train=r2_score(y_train,y_train_pred)
r2_test=r2_score(y_test,y_test_pred)
print(r2_train,r2_test)
#generate new data
x_range=np.linspace(40,90,300).reshape(-1,1)
y_range_pred=lr1.predict(x_range)
fig2=plt.figure()
plt.plot(x_range,y_range_pred)
plt.scatter(x_train,y_train)
plt.title=('predition')
plt.xlabel=('temperature')
plt.ylabel=('rate')
plt.show()
#generat new feature
from sklearn.preprocessing import PolynomialFeatures
poly2=PolynomialFeatures(degree=2)
x_2_train=poly2.fit_transform(x_train)
x_2_test=poly2.fit_transform(x_test)
print(x_2_train,x_2_test)
lr2=LinearRegression()
lr2.fit(x_2_train,y_train)
y_2_train_pred=lr2.predict(x_2_train)
y_2_test_pred=lr2.predict(x_2_test)
r2_2_train=r2_score(y_train,y_2_train_pred)
r2_2_test=r2_score(y_test,y_2_test_pred)
print('training r2_2:',r2_2_train)
print('test r2_2:',r2_2_test)
x_2_range=np.linspace(40,90,300).reshape(-1,1)
x_2_range=poly2.fit_transform(x_2_range)
y_2_range_pred=lr2.predict(x_2_range)
fig3=plt.figure()
plt.plot(x_range,y_2_range_pred)
plt.scatter(x_train,y_train)
plt.title=('predition')
plt.xlabel=('temperature')
plt.ylabel=('rate')
plt.show()
fig4=plt.figure()
plt.plot(x_range,y_2_range_pred)
plt.scatter(x_test,y_test)
plt.title=('predition')
plt.xlabel=('temperature')
plt.ylabel=('rate')
plt.show()
#观察过拟合
poly5=PolynomialFeatures(degree=5)
x_5_train=poly5.fit_transform(x_train)
x_5_test=poly5.fit_transform(x_test)
print(x_5_train,x_5_test)
lr5=LinearRegression()
lr5.fit(x_5_train,y_train)
y_5_train_pred=lr5.predict(x_5_train)
y_5_test_pred=lr5.predict(x_5_test)
r2_5_train=r2_score(y_train,y_5_train_pred)
r2_5_test=r2_score(y_test,y_5_test_pred)
print('training r2_5:',r2_5_train)
print('test r2_5:',r2_5_test)
x_5_range=np.linspace(40,90,300).reshape(-1,1)
x_5_range=poly5.fit_transform(x_5_range)
y_5_range_pred=lr5.predict(x_5_range)
fig5=plt.figure()
plt.plot(x_range,y_5_range_pred)
plt.scatter(x_train,y_train)
plt.scatter(x_test,y_test)
plt.title=('predition')
plt.xlabel=('temperature')
plt.ylabel=('rate')
plt.show()