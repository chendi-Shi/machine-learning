import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

# 读取数据
df = pd.read_excel(r"C:\Users\师晨迪\Documents\机器学习\机器学习\exam实战.xlsx")
print(df.to_string())

# 第一个图：所有点
fig = plt.figure()
plt.scatter(df.loc[:, 'exam1'], df.loc[:, 'exam2'])
plt.title('exam1-exam2')
plt.xlabel('exam1')
plt.ylabel('exam2')
plt.show()

# 创建mask
mask = df.loc[:, 'pass'] == 1
print(mask)

# 第二个图：区分通过和未通过
fig2 = plt.figure()

# 修正：在scatter中添加label参数
passed = plt.scatter(df.loc[:, 'exam1'][mask], df.loc[:, 'exam2'][mask],
                     color='green', marker='o', label='passed')
failed = plt.scatter(df.loc[:, 'exam1'][~mask], df.loc[:, 'exam2'][~mask],
                     color='red', marker='x', label='failed')

plt.title('exam1-exam2')
plt.xlabel('exam1')
plt.ylabel('exam2')

# 修正：使用正确的legend调用方式
plt.legend()  # 自动使用scatter中的label

plt.show()

# 继续你的其他代码...
x = df.drop('pass', axis=1)
print(x.head())
y = df.loc[:, 'pass']
print(y.head())

x1 = df.loc[:, 'exam1']
x2 = df.loc[:, 'exam2']
print(x.shape)
print(y.shape)

LR = LogisticRegression()
LR.fit(x, y)
y_predict = LR.predict(x)
print(y_predict)

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y, y_predict)
print(accuracy)

y_test = LR.predict([[70, 66]])
print('passed' if y_test == 1 else 'failed')
print(LR.coef_)
print(LR.intercept_)

theta0 = LR.intercept_
theta1, theta2 = LR.coef_[0][0], LR.coef_[0][1]
print(theta0, theta1, theta2)

x2_new = -(theta0 + theta1 * x1) / theta2  # 注意：这里应该有个负号

print(x2_new)

# 第三个图：决策边界
fig3 = plt.figure()
plt.plot(x1, x2_new, 'b-', label='Decision Boundary')

# 修正：在scatter中添加label
passed = plt.scatter(df.loc[:, 'exam1'][mask], df.loc[:, 'exam2'][mask],
                     color='green', marker='o', label='passed')
failed = plt.scatter(df.loc[:, 'exam1'][~mask], df.loc[:, 'exam2'][~mask],
                     color='red', marker='x', label='failed')

plt.title('exam1-exam2')
plt.xlabel('exam1')
plt.ylabel('exam2')
plt.legend()  # 修正：直接调用
plt.show()

# 多项式特征
x1_2 = x1 * x1
x2_2 = x2 * x2
x1_x2 = x1 * x2

# 修正：特征命名更清晰
x_new = pd.DataFrame({
    'x1': x1,
    'x2': x2,
    'x1_2': x1_2,
    'x2_2': x2_2,
    'x1_x2': x1_x2
})

LR2 = LogisticRegression()
LR2.fit(x_new, y)
y2_predict = LR2.predict(x_new)
accuracy2 = LR2.score(x_new, y)
print("二次逻辑回归准确率:", accuracy2)
print("系数:", LR2.coef_)
print("截距:", LR2.intercept_)

# 提取系数
coefs = LR2.coef_[0]
theta0 = LR2.intercept_[0]
theta1, theta2, theta3, theta4, theta5 = coefs[0], coefs[1], coefs[2], coefs[3], coefs[4]

print(f"theta0={theta0}, theta1={theta1}, theta2={theta2}, theta3={theta3}, theta4={theta4}, theta5={theta5}")

# 排序x1值用于绘图
x1_new = x1.sort_values()

# 计算二次决策边界
a = theta4  # x2^2的系数
b = theta5 * x1_new + theta2  # x2的系数
c = theta1 * x1_new + theta3 * x1_new * x1_new + theta0  # 常数项

def safe_sqrt(x):
    x_safe = np.maximum(x, 0)
    return np.sqrt(x_safe)

# 使用二次公式计算边界
discriminant = b**2 - 4*a*c
x2_new_boundary = (-b + safe_sqrt(discriminant)) / (2*a)

# 绘制二次决策边界
fig4 = plt.figure()
plt.plot(x1_new, x2_new_boundary, 'b-', label='Quadratic Decision Boundary')
passed = plt.scatter(df.loc[:, 'exam1'][mask], df.loc[:, 'exam2'][mask],
                     color='green', marker='o', label='passed')
failed = plt.scatter(df.loc[:, 'exam1'][~mask], df.loc[:, 'exam2'][~mask],
                     color='red', marker='x', label='failed')
plt.title('exam1-exam2 with Quadratic Boundary')
plt.xlabel('exam1')
plt.ylabel('exam2')
plt.legend()
plt.show()

# 计算两个边界（二次方程有两个解）
def f(x):
    a = theta4
    b = theta5 * x + theta2
    c = theta1 * x + theta3 * x * x + theta0
    discriminant = b**2 - 4*a*c
    x2_boundary1 = (-b + safe_sqrt(discriminant)) / (2*a)
    x2_boundary2 = (-b - safe_sqrt(discriminant)) / (2*a)
    return x2_boundary1, x2_boundary2

# 计算边界点
x2_boundary1 = []
x2_boundary2 = []

for x_val in x1_new:
    b1, b2 = f(x_val)
    x2_boundary1.append(b1)
    x2_boundary2.append(b2)

# 绘制两个边界
fig5 = plt.figure()
plt.plot(x1_new, x2_boundary1, 'b--', label='Boundary 1')
plt.plot(x1_new, x2_boundary2, 'b--', label='Boundary 2')
passed = plt.scatter(df.loc[:, 'exam1'][mask], df.loc[:, 'exam2'][mask],
                     color='green', marker='o', label='passed')
failed = plt.scatter(df.loc[:, 'exam1'][~mask], df.loc[:, 'exam2'][~mask],
                     color='red', marker='x', label='failed')
plt.title('exam1-exam2 with Both Boundaries')
plt.xlabel('exam1')
plt.ylabel('exam2')
plt.legend()
