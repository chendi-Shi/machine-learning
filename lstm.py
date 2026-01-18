#load the data
data=open(r'C:\Users\师晨迪\Documents\机器学习\lstm.txt').read()
#移除换行符
data=data.replace('\n',' ').replace('\r','')
print(data)
letters=list(set(data))
print(letters)
num_letters=len(letters)
print(num_letters)
#建立字典
int_to_char={a:b for a,b in enumerate(letters)}
print(int_to_char)
#char  to int
char_to_int={b:a for a,b in enumerate(letters)}
print(char_to_int)
import numpy as np
from tensorflow.keras.utils import to_categorical
#滑动窗口读取数据
def extract_data(data,slide):
    x=[]
    y=[]
    for i in range(len(data)-slide):
        x.append([a for a in data[i:i+slide]])
        y.append(data[i+slide])  # 修改1：只取下一个字符
    return x,y
#字符到数字的批量转换
def char_to_int_Data(x,y,char_to_int):
    x_to_int=[]
    y_to_int=[]
    for i in range(len(x)):
        x_to_int.append([char_to_int[char] for char in x[i]])
        y_to_int.append(char_to_int[y[i]])  # 修改2：y现在是单个字符，不是列表
    return x_to_int,y_to_int
def data_preprocessing(data,slide,num_letters,char_to_int):
    char_Data=extract_data(data,slide)
    int_Data=char_to_int_Data(char_Data[0],char_Data[1],char_to_int)
    Input=int_Data[0]
    Output=int_Data[1]  # 修改3：不需要flatten
    Input_RESHAPED=np.reshape(Input,(len(Input),slide))
    new=np.zeros((Input_RESHAPED.shape[0],Input_RESHAPED.shape[1],num_letters))  # 修改4：使用zeros
    for i in range(Input_RESHAPED.shape[0]):
        for j in range(Input_RESHAPED.shape[1]):
            new[i,j,:]=to_categorical(Input_RESHAPED[i,j],num_classes=num_letters)
    return new,Output
#extract x and y from text data
time_step = 20
x,y=data_preprocessing(data,time_step,num_letters,char_to_int)
print(x.shape)
print(len(y))
#split the data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.1,random_state=10)
print(x_train.shape,len(y_train))
y_train_category=to_categorical(y_train,num_letters)
print(y_train_category)
#set up the model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LSTM
model = Sequential()
model.add(LSTM(units=20,input_shape=(x_train.shape[1],x_train.shape[2]),activation='relu'))
model.add(Dense(num_letters,activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()
#train the model
model.fit(x_train,y_train_category,epochs=50,batch_size=100)
#make prediction based on the training data
y_train_pred=model.predict(x_train)
print(y_train_pred)
#transform the int to letters
y_train_pred_class = np.argmax(y_train_pred, axis=1)
y_train_pred_char=[int_to_char[i] for i in y_train_pred_class]
print(y_train_pred_char)
from sklearn.metrics import accuracy_score
accuracy_train=accuracy_score(y_train,y_train_pred_class)
print(accuracy_train)
y_test_pred=model.predict(x_test)
y_test_pred_class = np.argmax(y_test_pred, axis=1)
y_test_pred_char=[int_to_char[i] for i in y_test_pred_class]
accuracy_test=accuracy_score(y_test,y_test_pred_class)
print(y_test_pred_char)
print(accuracy_test)
new_letters='flare is a teacher who obtained his hpd in Australia'
x_new,y_new=data_preprocessing(data,time_step,num_letters,char_to_int)
y_new_pred=model.predict(x_new)
print(y_new_pred)
y_new_pred_indices= np.argmax(y_new_pred, axis=1)
y_new_pred_char= [int_to_char.get(int(i),'?')for i in y_new_pred_indices.flatten()]
print(y_new_pred_char)