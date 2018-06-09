# This code trains a CNN model to predict Arabic handwritten characters
# packages should be installed: numpy, pandas, keras, sklearn, and maybe tensorflow

#%%
#import packages
import numpy as np
import pandas as pd
#import tensorflow as tf
from keras.utils import to_categorical


#%%
#read data
folder = "C:/Users/user/Desktop/Handwritten/" #select ur directory
original_trainx = pd.read_csv( folder+"csvTrainImages 13440x1024.csv",header=None)
original_trainy = pd.read_csv(folder+"csvTrainLabel 13440x1.csv",header=None)
original_testx = pd.read_csv(folder+"csvTestImages 3360x1024.csv",header=None)
original_testy = pd.read_csv(folder+"csvTestLabel 3360x1.csv",header=None)


#%%
#images normalization
trainx = original_trainx.values.astype('float32')/255
testx = original_testx.values.astype('float32')/255

#labels zero-based numbering
trainy = original_trainy.values.astype('int32')-1
testy = original_testy.values.astype('int32')-1


#%%
#reshape 2d
trainx_2D = trainx.reshape([-1, 32, 32, 1])
testx_2D = testx.reshape([-1, 32, 32, 1])

#one hot encoding
original_trainy = trainy
trainy_1H = to_categorical(trainy, num_classes=28)
testy_1H = to_categorical(testy,28)


#%%
#CNN model building

#import keras
from keras.models import Sequential
from keras.layers import Dense , Flatten , Conv2D , MaxPooling2D # ,Dropout

input_size = trainx_2D.shape[1:]

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu', #border_mode="same", 
                 input_shape=input_size))
model.add(Conv2D(64 , kernel_size = (5 , 5) , activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128 , activation = 'relu'))
#model.add(Dropout(0.5))
model.add(Dense(28 , activation = 'softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer= 'adam',
              metrics=['accuracy'])


#%% CNN model training
model_info = model.fit(trainx_2D, trainy_1H,batch_size=256,epochs=5,verbose=1)
#model_info = model.fit(trainx_2D, trainy_1H,batch_size=256,epochs=15,verbose=0)


#%%
#model predictions
y_pred = model.predict_proba(testx_2D).ravel()
y_pred_1H = y_pred.reshape([3360,28])

y_pred_1H_reg = np.zeros(y_pred_1H.shape)
y_pred_1H_reg[y_pred_1H>0.5]=1

#convert 1 hot encoding back to single numeric 
y_pred_reg_numeric = [ np.argmax(t) for t in y_pred_1H_reg ]


#%%
#model evaluation
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, precision_score, recall_score

#confusion matrix
conf_mat = confusion_matrix(testy, y_pred_reg_numeric)

# accuracy
print("accuracy: ",accuracy_score(testy_1H, y_pred_1H_reg))
#precision
print("precision matrix: ",precision_score(testy_1H, y_pred_1H_reg, average=None))
print("average precision: ",precision_score(testy_1H, y_pred_1H_reg, average="micro"))
#recall
print("recall matrix: ",recall_score(testy_1H, y_pred_1H_reg, average=None))  
print("average recall: ",recall_score(testy_1H, y_pred_1H_reg, average="micro"))  
# f1
print("f1 matrix: ",f1_score(testy_1H, y_pred_1H_reg, average=None))
print("average f1: ",f1_score(testy_1H, y_pred_1H_reg, average="micro"))
