# import necessary function library
import os
import numpy as np
import random as random
from matplotlib import pyplot as plt
from PIL import Image
import gc
import time
import tensorflow as tf
from keras.layers import (Activation, Conv2D, Dense, Dropout, Flatten,BatchNormalization,
                          GlobalAveragePooling2D, Input, MaxPooling2D, add,concatenate)
from keras import Model
from keras.applications.resnet50 import ResNet50
import pandas as pd

# Factor decomposition funcitons

from FN import Estimate_FactorX
from FN import Separate_FactorX

'''
SETP:1 Load Data and Facotr Decomposition
'''

# 读取
cifar10 = tf.keras.datasets.cifar10
(X0, Y0), (X1, Y1) = cifar10.load_data()
# 整理格式
N0=X0.shape[0];N1=X1.shape[0]
X0 = X0.reshape(N0,32,32,3)/255.0
X1 = X1.reshape(N1,32,32,3)/255.0
X0 = X0.astype(np.float32)
X1 = X1.astype(np.float32)


X0 = X0.reshape(N0,3072)
X1 = X1.reshape(N1,3072)

# Estimate factor loading and regression coefficient
fweight ,beta = Estimate_FactorX(X0[0:1000])

# Factor decomposition
X0_new, weight0 = Separate_FactorX(X0,fweight,beta)
X1_new, weight1 = Separate_FactorX(X1,fweight,beta)

# Separate residual and factor
X00 = X0_new[:,1:3073];Z0 = X0_new[:,0:1]
X11 = X1_new[:,1:3073];Z1 = X1_new[:,0:1]

# Reshape the residual into image shape
X00 = tf.reshape(X00,[N0,32,32,3])
X11 = tf.reshape(X11,[N1,32,32,3])


'''
SETP:2 Model Reconstruction
'''

def fn_model(old_model,num_class):
    
    # Remove the last layer of old model
    
    num_layers =  len(old_model.layers)    
    print("Old model has " + str(num_layers) + " layers in total! ")
    model0 = Model(inputs=old_model.input, outputs=old_model.layers[num_layers-2].output)
    
    # Concatenate factor input with dense layer of old model
    input_z = Input([1])
    output_z = input_z
    model1 = Model(input_z, output_z)
    
    combined = concatenate([model0.output, model1.output])
    #combined = BatchNormalization()(combined)
    
    z = Dense(num_class, activation="softmax")(combined)
    
    new_model = Model(inputs=[model0.input, model1.input], outputs=z)
    
    return new_model


'''
 STEP3: Training FN 
'''

IMSIZE = 32
input_layer = Input([IMSIZE,IMSIZE,3])
x = input_layer
x = Conv2D(96,kernel_size=[3,3],padding='same', activation="relu")(x) 
x = MaxPooling2D([3,3], strides = [2,2])(x)    
x = Conv2D(256,kernel_size=[3,3], padding='same', activation="relu")(x)
x = MaxPooling2D([3,3], strides = [2,2])(x)
x = Conv2D(384,kernel_size=[3,3], padding='same', activation="relu")(x) 
x = Conv2D(384,kernel_size=[3,3], padding='same', activation="relu")(x) 
x = Conv2D(256,kernel_size=[3,3], padding='same', activation="relu")(x) 
x = MaxPooling2D([3,3], strides = [2,2])(x)
x = Flatten()(x)   
x = Dense(4096,activation = 'relu')(x)
x = Dropout(0.5)(x)
x = Dense(4096,activation = 'relu')(x)
x = Dropout(0.5)(x)
x = Dense(10,activation = 'softmax')(x) 
output_layer=x
model=Model(input_layer,output_layer)
model1 = fn_model(model,10)

batch_size = 200
K1 = int(np.ceil(N0/batch_size))
K2 = int(np.ceil(N1/batch_size))
B = 300

time_cost2 = np.zeros(B)
time_checker = 0


train_loss2 = np.zeros(B)
test_loss2 = np.zeros(B)

train_acc2 = np.zeros(B)
test_acc2= np.zeros(B)

optimizer = tf.keras.optimizers.SGD(learning_rate=0.001, momentum = 0.9, nesterov = True)
metrics = tf.keras.metrics.SparseCategoricalAccuracy()

# Adaptive learning rate for last layer
Xw = tf.concat([np.median(weight0[1:3073].numpy())/weight0[0] * tf.ones([4096,1]),tf.ones([1,1])],0)
bw = tf.reduce_mean(weight0[3073])/weight0[0]

for i in range(B):
    print("Current Epochs:",i+1)
    train_loss_avg = 0.
    test_loss_avg = 0.
    train_acc_avg = 0.
    test_acc_avg = 0.
    

    # Update whole epoch
    start = time.time()
    for k in range(K1):
        
        X,Y = X00[k*batch_size:(k+1)*batch_size,],Y0[k*batch_size:(k+1)*batch_size].reshape(batch_size,)
        Z = Z0[k*batch_size:(k+1)*batch_size,]
        with tf.GradientTape() as tape:
            Y_pred = model1([X,Z])
            loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=Y, y_pred=Y_pred)
            loss = tf.reduce_mean(loss)
        grads = tape.gradient(loss, model1.trainable_variables)
        l = len(grads)
        grads[l-2] = Xw * grads[l-2]; grads[l-1] = bw * grads[l-1]
        optimizer.apply_gradients(grads_and_vars=zip(grads, model1.trainable_variables))
    end = time.time()
    time_checker = time_checker + end-start
    time_cost2[i] = time_checker
    
    # Calculate training loss and accuracy
    for k in range(K1):
        X,Y = X00[k*batch_size:(k+1)*batch_size,],Y0[k*batch_size:(k+1)*batch_size].reshape(batch_size,)
        Z = Z0[k*batch_size:(k+1)*batch_size,]
        #Z  = (Z-np.mean(Z))/(np.std(Z)+1e-6)
        Y_pred = model1([X,Z])                                                         
        loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=Y, y_pred=Y_pred)   
        loss = tf.reduce_mean(loss)                                                
        train_loss_avg = train_loss_avg + loss.numpy()
        metrics.update_state(y_true=Y, y_pred=Y_pred)
        train_acc_avg = train_acc_avg + metrics.result().numpy()    
    
    train_loss2[i] = train_loss_avg/K1
    train_acc2[i] = train_acc_avg/K1
    
    
    # Calculate test loss and accuracy
    for k in range(K2):
        X,Y = X11[k*batch_size:(k+1)*batch_size,],Y1[k*batch_size:(k+1)*batch_size].reshape(batch_size,)
        Z = Z1[k*batch_size:(k+1)*batch_size,]
        Y_pred = model1([X,Z])                                                          
        loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=Y, y_pred=Y_pred)   
        loss = tf.reduce_mean(loss)                                                
        test_loss_avg = test_loss_avg + loss.numpy()
        metrics.update_state(y_true=Y, y_pred=Y_pred)
        test_acc_avg = test_acc_avg + metrics.result().numpy()    
    
    test_loss2[i] = test_loss_avg/K2
    test_acc2[i] = test_acc_avg/K2
    
    
df = pd.DataFrame([time_cost2,train_loss2,train_acc2,test_loss2,test_acc2])
new_col = ['time_cost','train_loss','train_acc','test_loss','test_acc']
df2 = pd.DataFrame(df.values.T, columns=new_col)
df2.to_csv('./AlexNet_CIFAR10_FN_NAG_0001.csv', index=False, header=True)

