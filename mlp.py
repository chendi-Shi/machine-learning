import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
model = Sequential()
#通过.add（）添加各层网络
from keras.layers import Dense
model.add(Dense(3,activation='sigmoid',input_dim=3))
model.add(Dense(3,activation='sigmoid'))
model.summary()
model.compile(loss='binary_crossentropy',optimizer='sgd',metrics=['accuracy'])
#model.fit(epochs=5)
data=pd.read_excel(r"C:\Users\师晨迪\Documents\机器学习\机器学习\mlp.xlsx")
print(data.head())
x=data.drop(['y'],axis=1)
y=data.loc[:,'y']
#visuaize the data
fig1=plt.figure()
passed=plt.scatter(x.loc[:,'x1'][y==1],x.loc[:,'x2'][y==1])
failed=plt.scatter(x.loc[:,'x1'][y==0],x.loc[:,'x2'][y==0])
plt.legend([passed,failed],['Passed','Failed'])
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('raw title')
plt.show()
#split the data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.33,random_state=10)
#set up the model
from keras.models import Sequential
from keras.layers import Dense,Activation
mlp=Sequential()
mlp.add(Dense(20,input_dim=2,activation='sigmoid'))
mlp.add(Dense(1,activation='sigmoid'))
mlp.summary()
mlp.compile(loss='binary_crossentropy',optimizer='adam')
#fit the model
mlp.fit(x_train,y_train,epochs=30)
y_train_predict=mlp.predict(x_train)
y_train_predict_class=(y_train_predict>0.5)
from sklearn.metrics import accuracy_score
accuracy_train=accuracy_score(y_train,y_train_predict_class)
print(accuracy_train)
#make prediction on the test data
y_test_predict=mlp.predict(x_test)
y_test_predict_classes=(y_test_predict>0.5)
accuracy_test=accuracy_score(y_test,y_test_predict_classes )
print(accuracy_test)
y_train_predict_form=pd.Series(i[0] for i in y_train_predict)
print(y_train_predict_form)
#generate new data for plot
xx,yy=np.meshgrid(np.arange(0,1,0.01),np.arange(0,1,0.01))
x_range=np.c_[xx.ravel(),yy.ravel()]
y_range_prob=mlp.predict(x_range)
y_range_predict=(y_range_prob>0.5).astype(int)
# 1. 生成网格预测
y_range_prob = mlp.predict(x_range)
y_range_predict = (y_range_prob > 0.5).astype(int)
print("网格预测形状:", y_range_predict.shape)

# 2. 创建Series - 关键修正：处理不同形状的数组
# 如果 y_range_predict 是 (n,1) 形状
if len(y_range_predict.shape) == 2 and y_range_predict.shape[1] == 1:
    y_range_series = pd.Series(y_range_predict.flatten())  # 展平为一维
else:
    y_range_series = pd.Series(y_range_predict)  # 已经是一维

print("Series前10个值:")
print(y_range_series.head(10))

# 3. 同样处理训练集预测（如果需要的话）
# 如果你确实需要 y_train_predict_form
y_train_predict = mlp.predict(x_train)  # 确保这个已定义
y_train_pred_class = (y_train_predict > 0.5).astype(int)

if len(y_train_pred_class.shape) == 2 and y_train_pred_class.shape[1] == 1:
    y_train_predict_form = pd.Series(y_train_pred_class.flatten())
else:
    y_train_predict_form = pd.Series(y_train_pred_class)

# 4. 绘图
fig2 = plt.figure(figsize=(10, 8))

# 确保 y_range_predict 是一维的用于索引
if len(y_range_predict.shape) == 2:
    y_range_flat = y_range_predict[:, 0]  # 取第一列
else:
    y_range_flat = y_range_predict

# 绘制网格预测点（抽样显示避免过多）
sample_size = min(1000, len(x_range))
indices = np.random.choice(len(x_range), sample_size, replace=False)

# 预测为1的点
mask1 = y_range_flat[indices] == 1
if np.any(mask1):
    passed_predict = plt.scatter(
        x_range[indices, 0][mask1],
        x_range[indices, 1][mask1],
        c='lightblue', marker='.', s=20, alpha=0.6, label='Predicted Pass'
    )

# 预测为0的点
mask0 = y_range_flat[indices] == 0
if np.any(mask0):
    failed_predict = plt.scatter(
        x_range[indices, 0][mask0],
        x_range[indices, 1][mask0],
        c='lightcoral', marker='.', s=20, alpha=0.6, label='Predicted Fail'
    )

# 原始数据点 - 修正语法错误
passed = plt.scatter(
    x.loc[:, 'x1'][y == 1],
    x.loc[:, 'x2'][y == 1],
    c='blue', marker='o', s=50, alpha=0.8, label='Actual Pass'
)

failed = plt.scatter(
    x.loc[:, 'x1'][y == 0],
    x.loc[:, 'x2'][y == 0],
    c='red', marker='x', s=50, alpha=0.8, label='Actual Fail'
)

# 修正图例语法
plt.legend()
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Prediction Result')
plt.grid(True, alpha=0.3)
plt.show()