import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
df=pd.read_excel(r"C:\Users\师晨迪\Documents\机器学习\机器学习\芯片实战.xlsx")
print(df.to_string())
mask=df.loc[:,'pass']==1
print(~mask)
fig2=plt.figure()
passed=plt.scatter(df.loc[:,'test1'][mask],df.loc[:,'test2'][mask])
failed=plt.scatter(df.loc[:,'test1'][~mask],df.loc[:,'test2'][~mask])
plt.title('test1-test2')
plt.xlabel('test1')
plt.ylabel('test2')
plt.legend((passed,failed),('passed','failed'))
plt.show()
x=df.drop('pass',axis=1)
print(x.head())
y=df.loc[:,'pass']
print(y.head())
x1=df.loc[:,'test1']
x2=df.loc[:,'test2']
x1_2=x1*x1
x2_2=x2*x2
x1_x2=x1*x2
x_new={'x1':x1,'x2':x2,'x1_2':x1_2,'x2_2':x2_2,'x1_x2':x1_x2}
x_new=pd.DataFrame(x_new)
print(x_new)
LR2=LogisticRegression()
LR2.fit(x_new,y)
y2_predict=LR2.predict(x_new)
accuracy2=LR2.score(x_new,y)
print(accuracy2)
x1_new=x1.sort_values()
theta1,theta2,theta3,theta4,theta5=LR2.coef_[0][0],LR2.coef_[0][1],LR2.coef_[0][2],LR2.coef_[0][3],LR2.coef_[0][4]
theta0=LR2.intercept_
x1_new
a=theta4
b=theta5*x1_new+theta2
c=theta1*x1_new+theta3*x1_new*x1_new+theta0
def safe_sqrt(x):
    x_safe=np.maximum(x,0)
    return np.sqrt(x_safe)
x2_new_boundary=(-b+safe_sqrt(b**2-4*a*c))/(2*a)
print(x2_new_boundary)
fig5=plt.figure()
plt.plot(x1_new,x2_new_boundary)
passed=plt.scatter(df.loc[:,'test1'][mask],df.loc[:,'test2'][mask])
failed=plt.scatter(df.loc[:,'test1'][~mask],df.loc[:,'test2'][~mask])
plt.title('test1-test2')
plt.xlabel('test1')
plt.ylabel('test2')
plt.legend((passed,failed),('passed','failed'))
plt.show()