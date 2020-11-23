#!/usr/bin/env python
# coding: utf-8

# In[8]:


import os
import numpy as np
import random as random
import gc
import time
import tensorflow as tf
from keras.layers import (Activation, Conv2D, Dense, Dropout, Flatten,BatchNormalization,
                          GlobalAveragePooling2D, Input, MaxPooling2D, add,concatenate)
from keras import Model
from keras.models import load_model
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import ResNet50

from FN import Estimate_FactorX
from FN import Separate_FactorX


# In[ ]:


train_path = '/home/pkustudent/notebooks/BASICS/Lecture 5.2 - Data Augumentation/data/CatDog/train/'
validation_path = '/home/pkustudent/notebooks/BASICS/Lecture 5.2 - Data Augumentation/data/CatDog/validation/'

N0 = 1500
N1 = 10000
IMSIZE = 224
Epochs = 300
batch_size1 = 75
batch_size2 = 100
K0 = int(N0/batch_size1)
K1 = int(N1/batch_size2)


# In[ ]:


train_generator = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.5,
        zoom_range=0.2, 
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True).flow_from_directory(
        train_path,
        target_size=(IMSIZE, IMSIZE),
        batch_size=N0,
        class_mode='categorical')


# In[7]:


model_fn = load_model("./resnet_fn.h5")


# In[ ]:


data = np.load("weight.npz")
fweight = data["fweight"]
beta = data["beta"]
s = data["s"]

ratio = (s[0]/(s[1]+1e-7))**2
Xw = tf.concat([tf.ones([2048,1]),(1/ratio) * tf.ones([1,1])],0)
#Xw = tf.concat([tf.ones([4096,1]),(1/ratio) * tf.ones([1,1])],0)


# In[ ]:


time_cost = np.zeros(Epochs)
time_checker = 0

loss = np.zeros(Epochs)
val_loss = np.zeros(Epochs)

acc = np.zeros(Epochs)
val_acc = np.zeros(Epochs)
    
    
metrics = tf.keras.metrics.CategoricalAccuracy()

for i in range(Epochs):
    
    cumulated_time = 0.
    cumulated_ss_train = 0
    cumulated_ss_val = 0
    optimizer = tf.keras.optimizers.SGD(learning_rate = 0.05/np.sqrt(i+1))
    train_loss_avg = 0.
    test_loss_avg = 0.
    train_acc_avg = 0.
    test_acc_avg = 0.
    
    for j in range(10):
    # 重新生成全样本
        X_old,Y0 = next(train_generator)
        X_old = tf.reshape(X_old,[N0,224*224*3])
        start = time.time()
        X0,Z0 = Separate_FactorX(X_old,fweight,beta)
        end1 = time.time()
        X0 = tf.reshape(X0,[N0,224,224,3])
    
    # 训练
        for k in range(K0):
            X= X0[k*batch_size1:(k+1)*batch_size1]
            Y = Y0[k*batch_size1:(k+1)*batch_size1]
            Z = Z0[k*batch_size1:(k+1)*batch_size1]
            ss = X.shape[0]
            with tf.GradientTape() as tape:
                Y_pred = model_fn([X,Z])
                Loss = tf.keras.losses.categorical_crossentropy(y_true=Y, y_pred=Y_pred)
                Loss = tf.reduce_mean(Loss)
            grads = tape.gradient(Loss, model_fn.trainable_variables)
            grads[len(grads)-2] = Xw * grads[len(grads)-2]
            #grads[len(grads)-2] = Xw * grads[len(grads)-2]; grads[len(grads)-1] = bw * grads[len(grads)-1]
            optimizer.apply_gradients(grads_and_vars=zip(grads, model_fn.trainable_variables))
        
            train_loss_avg = (cumulated_ss_train * train_loss_avg + ss * Loss.numpy())/(ss+cumulated_ss_train)
            
            metrics.reset_states()
            metrics.update_state(y_true=Y, y_pred=Y_pred)
            train_acc_avg =   (cumulated_ss_train * train_acc_avg + ss * metrics.result().numpy())/(ss+cumulated_ss_train)
            
            cumulated_ss_train = cumulated_ss_train + ss
            print("\rEpoch: {:d} batch: {:d} loss: {:.4f} acc: {:.4f} sep_time: {:.2f}| {:.2%}"
                .format(i+1, j*K0+k+1 , train_loss_avg, train_acc_avg, end1-start ,(j*K0+k+1)*1.0/(15*K0)), end='',  flush=True)
            #print(end-start)
        end = time.time()
        cumulated_time = cumulated_time + end-start
    
    time_checker = time_checker + cumulated_time
    time_cost[i] = time_checker
 
    loss[i] = train_loss_avg
    acc[i] = train_acc_avg
    
    # 进行验证
    for k in range(K1):
        data = np.load("./validation/"+str(k)+".npz")
        X = data["X11"]
        Y = data["Y1"]
        Z = data["Z1"]
        ss = X.shape[0]
        Y_pred = model_fn([X,Z])                                                          # 通过调用model，而不是显式表达
        Loss = tf.keras.losses.categorical_crossentropy(y_true=Y, y_pred=Y_pred)   # 损失采用cross_entropy，是一个ss*1的向量
        Loss = tf.reduce_mean(Loss)                                                # 每一维求均值得到均值版本的loss
        test_loss_avg = (cumulated_ss_val * test_loss_avg + ss * Loss.numpy())/(ss+cumulated_ss_val)
        metrics.reset_states()
        metrics.update_state(y_true=Y, y_pred=Y_pred)
        test_acc_avg =   (cumulated_ss_val* test_acc_avg + ss * metrics.result().numpy())/(ss+cumulated_ss_val) 
        cumulated_ss_val = cumulated_ss_val + ss
    val_loss[i] = test_loss_avg
    val_acc[i] = test_acc_avg
    
    print("\rEpoch: {:d}/{:d} | loss: {:.4f} acc: {:.4f} | val_loss: {:.4f}  val_acc: {:.4f} | time: {:.2f}s"
            .format(i+1, Epochs, loss[i], acc[i], val_loss[i], val_acc[i], cumulated_time), end='\n')


