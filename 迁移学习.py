from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
path=r"C:\Users\师晨迪\Documents\机器学习\机器学习\data(1)\data\original_data"
dst_path=r"C:\Users\师晨迪\Documents\机器学习\机器学习\data(1)\data\original_data\gen_data"
datagen=ImageDataGenerator(rotation_range=10,width_shift_range=0.1,height_shift_range=0.2,horizontal_flip=True)
gen=datagen.flow_from_directory(path,target_size=(224,224),batch_size=2,save_prefix='gen',save_format='jpg')
for i in range(100):
    next(gen)
#load the image
from tensorflow.keras.preprocessing.image import load_img,img_to_array
img_path=r"C:\Users\师晨迪\Documents\机器学习\机器学习\data(1)\data\test\1.jpg"
img=load_img(img_path,target_size=(224,224))
type(img)
fig1=plt.figure()
plt.imshow(img)
plt.show()
img=img_to_array(img)
type(img)
print(type(img))
print(img.shape)
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import VGG16
model_vgg=VGG16(weights='imagenet',include_top=False)
x=np.expand_dims(img,axis=0)
x=preprocess_input(x)
print(x.shape)
features=model_vgg.predict(x)
print(features.shape)
#flatten
features=features.reshape(1,7*7*512)
print(features.shape)
#list all the names of the data
import os
folder=r"C:\Users\师晨迪\Documents\机器学习\机器学习\data(1)\data\original_data"
dirs=os.listdir(folder)
#名称合并
img_path=[]
for i in dirs:
    if os.path.splitext(i)[1]=='.jpg':
        img_path.append(i)
img_path=[folder+'/'+i for i in img_path]
print(img_path)
#define a method to extract the features
def modelprocess(img_path,model_vgg):
    img=load_img(img_path,target_size=(224,224))
    img=img_to_array(img)
    x=np.expand_dims(img,axis=0)
    x_VGG=model_vgg.predict(x)
    x_VGG=x_VGG.reshape(1,7*7*512)
    return x_VGG
#图像批量处理
features_train=np.zeros([len(img_path),7*7*512])
for i in range(len(img_path)):
    features_i=modelprocess(img_path[i],model_vgg)
    print('preprocessed:',img_path[i])
    features_train[i]=features_i
x=features_train
#set up a  kmeans_un_supervised model
from sklearn.cluster import KMeans
cnn_kmeans=KMeans(n_clusters=2,max_iter=1000)
cnn_kmeans.fit(x)
#make prediction
y_pred_kmeans=cnn_kmeans.predict(x)
print(y_pred_kmeans)
from collections import Counter
print(Counter(y_pred_kmeans))
normal_apple_id=1
#visualize the result
fig2=plt.figure(figsize=(10,40))
for i in range(30):
    for j in range(5):
        img=load_img(img_path[i*5+j] if i*5+j<len(img_path) else r"C:\Users\师晨迪\Documents\机器学习\机器学习\data(1)\data\original_data\4-0.jpg")#read the image
        plt.subplot(30,5,i*5+j+1)
        plt.title('apple' if y_pred_kmeans[i]==normal_apple_id else 'others')
        plt.imshow(img),plt.axis('off')
plt.show()