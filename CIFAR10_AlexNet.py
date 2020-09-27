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
SETP:1 Load Data
'''

# Load from Keras
cifar10 = tf.keras.datasets.cifar10
(X0, Y0), (X1, Y1) = cifar10.load_data()

N0=X0.shape[0];N1=X1.shape[0]
X0 = X0.reshape(N0,32,32,3)/255.0
X1 = X1.reshape(N1,32,32,3)/255.0
X0 = X0.astype(np.float32)
X1 = X1.astype(np.float32)


'''
SETP:2 Build AlexNet
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


'''
SETP:3  Optimization
'''

batch_size = 200
K1 = int(np.ceil(N0/batch_size))
K2 = int(np.ceil(N1/batch_size))
B = 300

time_cost = np.zeros(B)
time_checker = 0

train_loss1 = np.zeros(B)
test_loss1 = np.zeros(B)

train_acc1 = np.zeros(B)
test_acc1 = np.zeros(B)

optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, momentum = 0.9, nesterov = True)
#optimizer = tf.keras.optimizers.Adagrad(learning_rate=0.001)
#optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
metrics = tf.keras.metrics.SparseCategoricalAccuracy()

# Main Loop
for i in range(B):
    print("Current Epochs:",i+1)
    train_loss_avg = 0.
    test_loss_avg = 0.
    train_acc_avg = 0.
    test_acc_avg = 0.
    
    # Update whole epoch
    start = time.time()
    for j in range(K1):
        X,y = X0[j*batch_size:(j+1)*batch_size,],Y0[j*batch_size:(j+1)*batch_size].reshape(batch_size,)
        with tf.GradientTape() as tape:
            y_pred = model(X)
            loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=y, y_pred=y_pred)
            loss = tf.reduce_mean(loss)
        grads = tape.gradient(loss, model.variables)
        optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))
    end = time.time()
    time_checker = time_checker + end-start
    time_cost[i] = time_checker
    
    # Calculate training loss and accuracy
    for k in range(K1):
        X,Y = X0[k*batch_size:(k+1)*batch_size,],Y0[k*batch_size:(k+1)*batch_size].reshape(batch_size,)
        Y_pred = model(X)                                                          
        loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=Y, y_pred=Y_pred)   
        loss = tf.reduce_mean(loss)                                                
        train_loss_avg = train_loss_avg + loss.numpy()
        metrics.update_state(y_true=Y, y_pred=Y_pred)
        train_acc_avg = train_acc_avg + metrics.result().numpy()    
    
    train_loss1[i] = train_loss_avg/K1
    train_acc1[i] = train_acc_avg/K1
    
    # Calculate test loss and accuracy
    for k in range(K2):
        X,Y = X1[k*batch_size:(k+1)*batch_size,],Y1[k*batch_size:(k+1)*batch_size].reshape(batch_size,)
        Y_pred = model(X)                                                          
        loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=Y, y_pred=Y_pred)   
        loss = tf.reduce_mean(loss)                                                
        test_loss_avg = test_loss_avg + loss.numpy()
        metrics.update_state(y_true=Y, y_pred=Y_pred)
        test_acc_avg = test_acc_avg + metrics.result().numpy()    
    
    test_loss1[i] = test_loss_avg/K2
    test_acc1[i] = test_acc_avg/K2
    
# Save data  
df = pd.DataFrame([time_cost,train_loss1,train_acc1,test_loss1,test_acc1])
new_col = ['time_cost','train_loss','train_acc','test_loss','test_acc']
df2 = pd.DataFrame(df.values.T, columns=new_col)
df2.to_csv('./AlexNet_CIFAR10_NAG_001.csv', index=False, header=True)
