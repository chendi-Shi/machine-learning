import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data=pd.read_excel(r"C:\Users\师晨迪\Documents\机器学习\data_class_raw.xlsx")
print(data.head())
x=data.drop(['y'],axis=1)
y=data.loc[:,'y']
#visualize the result
fig1=plt.figure()
bad=plt.scatter(x.loc[:,'x1'][y==0],x.loc[:,'x2'][y==0])
good=plt.scatter(x.loc[:,'x1'][y==1],x.loc[:,'x2'][y==1])
plt.legend((bad,good),('bad','good'))
plt.title=('raw data')
plt.xlabel=('x1')
plt.ylabel=('x2')
plt.show()
#anomy detection
from sklearn.covariance import EllipticEnvelope
ad_model=EllipticEnvelope(contamination=0.02)
ad_model.fit(x[y==0])
y_pred_bad=ad_model.predict(x[y==0])
print(y_pred_bad)
#visualize the anomly result
fig2=plt.figure()
bad=plt.scatter(x.loc[:,'x1'][y==0],x.loc[:,'x2'][y==0])
good=plt.scatter(x.loc[:,'x1'][y==1],x.loc[:,'x2'][y==1])
plt.scatter(x.loc[:,'x1'][y==0][y_pred_bad==-1],x.loc[:,'x2'][y==0][y_pred_bad==-1],color='red',marker='x',s=150)
plt.legend((bad,good),('bad','good'))
plt.title=('raw data')
plt.xlabel=('x1')
plt.ylabel=('x2')
plt.show()
data=pd.read_excel(r"C:\Users\师晨迪\Documents\机器学习\data_class_processed.xlsx")
print(data.head())
x=data.drop(['y'],axis=1)
y=data.loc[:,'y']
#pca
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
x_norm=StandardScaler().fit_transform(x)
pca=PCA(n_components=2)
x_reduced=pca.fit_transform(x_norm)
var_ratio=pca.explained_variance_ratio_
print(var_ratio)
fig3=plt.figure()
plt.bar([1,2],var_ratio)
plt.show()
#train and test split:random_state=4,test_size=0.4
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.4,random_state=4)
print(x_train.shape,y_train.shape,x_test.shape,y_test.shape)
from sklearn.neighbors import KNeighborsClassifier
knn_10=KNeighborsClassifier(n_neighbors=10)
knn_10.fit(x_train,y_train)
y_train_predict=knn_10.predict(x_train)
y_test_predict=knn_10.predict(x_test)
#calculate the accuracy
from sklearn.metrics import accuracy_score
acuracy_train=accuracy_score(y_train,y_train_predict)
acuracy_test=accuracy_score(y_test,y_test_predict)
print('training score:',acuracy_train)
print('test score:',acuracy_test)
#visualize the knn result and the boundary
xx,yy=np.meshgrid(np.arange(0,10,0.05),np.arange(0,10,0.05))
x_range=np.c_[xx.ravel(),yy.ravel()]
y_range_pred=knn_10.predict(x_range)
fig4=plt.figure()
knn_bad=plt.scatter(x_range[:,0][y_range_pred==0],x_range[:,1][y_range_pred==0])
knn_good=plt.scatter(x_range[:,0][y_range_pred==1],x_range[:,1][y_range_pred==1])
bad=plt.scatter(x.loc[:,'x1'][y==0],x.loc[:,'x2'][y==0])
good=plt.scatter(x.loc[:,'x1'][y==1],x.loc[:,'x2'][y==1])
plt.legend((bad,good),('bad','good'))
plt.title=('prediction result')
plt.xlabel=('x1')
plt.ylabel=('x2')
plt.show()
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_test_predict)
print(cm)
TP=cm[1,1]
TN=cm[0,0]
FP=cm[0,1]
FN=cm[1,0]
print(TP,TN,FP,FN)
accuracy=(TP+TN)/(TP+TN+FP+FN)
print(accuracy)
recall=TP/(TP+FN)
print(recall)
precision=TP/(TP+FP)
print(precision)
Specificity=TN/(TP+FN)
print(Specificity)
f1=2*(recall*precision)/(recall+precision)
print(f1)
#try differeent k and calculate the result
n=[i for i in range(1,21)]
accuracy_train=[]
accuracy_test=[]
for i in n:
    knn=KNeighborsClassifier(n_neighbors=i)
    knn.fit(x_train,y_train)
    y_train_predict=knn.predict(x_train)
    y_test_predict=knn.predict(x_test)
    accuracy_train_i=accuracy_score(y_train,y_train_predict)
    accuracy_test_i=accuracy_score(y_test,y_test_predict)
    accuracy_train.append(accuracy_train_i)
    accuracy_test.append(accuracy_test_i)
print(accuracy_train)
print(accuracy_test)
fig5=plt.figure()
plt.subplot(121)
plt.plot(n,accuracy_train,marker='o')
plt.title=('train accuracy')
plt.xlabel=('n')
plt.ylabel=('accuracy')
plt.subplot(122)
plt.plot(n,accuracy_test,marker='x')
plt.show()
