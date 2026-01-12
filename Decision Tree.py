import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data=pd.read_excel(r"C:\Users\师晨迪\Documents\机器学习\机器学习\PCA决策树实战.xlsx")
print(data.to_string())
x=data.drop(['target','label'],axis=1)
y=data['target']
print(y.head())
print(x.shape,y.shape)
#establish the decision tree model
from sklearn import tree
dc_tree=tree.DecisionTreeClassifier(criterion='entropy',min_samples_leaf=5)
dc_tree.fit(x,y)
y_pred=dc_tree.predict(x)
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y,y_pred)
print(accuracy)
#visualize the tree
fig=plt.figure()
tree.plot_tree(dc_tree,filled=True,feature_names=['sepal length', 'sepal width','petal length','petal width']
,class_names=['setosa','versicolor','virginica'])
plt.show()
