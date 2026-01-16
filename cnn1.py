from tensorflow.keras.preprocessing.image import ImageDataGenerator
train_generator=ImageDataGenerator(rescale=1./255)
training_set=train_generator.flow_from_directory(r'C:\Users\师晨迪\Documents\机器学习\机器学习\archive\training_set',target_size=(50,50),batch_size=32,class_mode='binary')
#SET UP THE CNN MODEL
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Flatten,Dense
model=Sequential()
#卷积层
model.add(Conv2D(32,(3,3),activation='relu',input_shape=(50,50,3)))
#池化层
model.add(MaxPooling2D(pool_size=(2,2)))
#卷积层
model.add(Conv2D(32,(3,3),activation='relu'))
#池化层
model.add(MaxPooling2D(pool_size=(2,2)))
#flattening layer
model.add(Flatten())
model.add(Dense(128,activation='relu'))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()
#train the model
model.fit(training_set,epochs=25)
#accuracy on the training data
accuracy_train=model.evaluate(training_set)
print(accuracy_train)
#accuracy on the test data
test_set=train_generator.flow_from_directory(r'C:\Users\师晨迪\Documents\机器学习\机器学习\archive\test_set',target_size=(50,50),batch_size=32,class_mode='binary')
accuracy_test=model.evaluate(test_set)
print(accuracy_test)
#load single image
from tensorflow.keras.preprocessing.image import load_img,img_to_array
pic_cat=r'C:\Users\师晨迪\Documents\机器学习\机器学习\archive\cat.4196.jpg'
pic_cat=load_img(pic_cat,target_size=(50,50))
pic_cat=img_to_array(pic_cat)
pic_cat=pic_cat/255.0
pic_cat=pic_cat.reshape(1,50,50,3)
result=model.predict(pic_cat)
print(result)
#make prediction on multiple images
import matplotlib.pyplot as plt
import matplotlib as mlp
font2={'family':'SimHei',
       'weight':'normal',
       'size':'20', }
plt.rcParams['font.family'] ='SimHei'
plt.rcParams['axes.unicode_minus']=False
image_dir=r'C:\Users\师晨迪\Documents\机器学习\机器学习\archive\test_set\test_set\cats'
from matplotlib.image import imread
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
a=list(range(4001,4010))
fig=plt.figure()
for i in a:
    img_name=f"{image_dir}\\cat.{i}"+'.jpg'
    img_ori=load_img(img_name,target_size=(50,50))
    img=img_to_array(img_ori)
    img=img.astype('float32')/255.0
    img=img.reshape(1,50,50,3)
    result=model.predict(img)
    img_ori=load_img(img_name,target_size=(250,250))
    plt.subplot(3,3,i-4000)
    plt.imshow(img_ori)
    plt.title('预测为：狗' if result[0][0]==1 else'预测为：猫')
plt.show()
