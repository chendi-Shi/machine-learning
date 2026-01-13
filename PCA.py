import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data=pd.read_excel(r"C:\Users\师晨迪\Documents\机器学习\机器学习\PCA决策树实战.xlsx")
print(data.to_string())
x=data.drop(['target','label'],axis=1)
y=data.loc[:,'label']
print(y.head())
#establish knn model and calculate the accuracy
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier()
knn.fit(x,y)
y_pred=knn.predict(x)
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y,y_pred)
print(accuracy)
from sklearn.preprocessing import StandardScaler
x_norm=StandardScaler().fit_transform(x)
print(x_norm)
fig1=plt.figure()
plt.subplot(121)
plt.hist(x.loc[:,'sepal length '],bins=100)
plt.subplot(122)
plt.hist(x_norm[:,0],bins=100)
plt.show()
x1_mean=x.loc[:,'sepal length '].mean()
x1_norm_mean=x.iloc[:,0]
x1_sigma=x.loc[:,'sepal length '].std()
x1_mean_sigma=x.iloc[:,0].std()
print({'x1_mean,x1_norm_mean,x1_sigma,x1_mean_sigma is'},x1_mean,x1_norm_mean,x1_sigma,x1_mean_sigma)
#pca analysis
from sklearn.decomposition import PCA
pca=PCA(n_components=4)
x_pca=pca.fit_transform(x_norm)
var_ratio=pca.explained_variance_ratio_
print(var_ratio)
#visualize the pca result
fig2=plt.figure()
plt.bar([1,2,3,4],var_ratio)
plt.xticks([1,2,3,4],['PC1','PC2','PC3','PC4'])
plt.ylabel('varience ratio of each component')
plt.show()
pca=PCA(n_components=2)
x_pca2=pca.fit_transform(x_norm)
var_ratio2=pca.explained_variance_ratio_
print(var_ratio2)
fig3=plt.figure()
setosa=plt.scatter(x_pca2[:,0][y==0],x_pca2[:,1][y==0])
ver=plt.scatter(x_pca2[:,0][y==1],x_pca2[:,1][y==1])
vir=plt.scatter(x_pca2[:,0][y==2],x_pca2[:,1][y==2])
plt.legend((setosa,ver,vir),('setosa','ver','vir'),loc='upper right')
plt.show()