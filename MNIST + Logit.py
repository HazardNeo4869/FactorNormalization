#!/usr/bin/env python
# coding: utf-8


'''
MNIST + Logit
'''

import numpy as np
import tensorflow as tf    #tf.compat.v1
import gc
import time
import pandas as pd

from keras.layers import Dense,Flatten, Input
from keras import Model
from matplotlib import pyplot as plt



'''
STEP.1: Load Data and Factor Decomposition 
'''
mnist = tf.keras.datasets.mnist
(X0, Y0), (X1, Y1) = mnist.load_data()

N0=X0.shape[0];N1=X1.shape[0]
X0 = X0.reshape(N0,784)/255.0
X1 = X1.reshape(N1,784)/255.0
Y0 = Y0.reshape(60000,1)
Y1 = Y1.reshape(10000,1)

mean = np.mean(X0,axis=(0,1))
std = np.std(X0,axis=(0,1))
X0 = (X0-mean)/(std+1e-7)
X1 = (X1-mean)/(std+1e-7)

X0 = X0.astype(np.float32)
X1 = X1.astype(np.float32)


'''
Estimate_FactorX
'''
def Estimate_FactorX(X,k):
    
    X = tf.constant(X, dtype = tf.float32)
    ss,ncov = X.shape
     
    # Add bias
    X = tf.concat([X,tf.ones([ss,1])],1)
    beta = tf.zeros(1)
    
    # SVD
    s,u,v = tf.compat.v1.svd(X,full_matrices=True)     
    fv = v[:,0:k]
    
    del u,v
    _ = gc.collect() 
    
    # calculate Top k factor
    xf = tf.reshape(tf.matmul(X,fv),[ss,k])
    
    # OLS for residual
    A = tf.compat.v1.matrix_inverse(tf.matmul(tf.transpose(xf),xf) + 1e-6);
    B = tf.matmul(tf.transpose(xf),X)
    beta = tf.matmul(A,B)
    
    del xf,A,B
    _ = gc.collect() #释放内存
    
    return fv,beta,s


'''
Separate_FactorX
'''
def Separate_FactorX(X,fweight,beta):
    
    k = fweight.shape[1]
    ss,ncov = X.shape
    # Add bias
    X = tf.concat([X,tf.ones([ss,1])],1)
    # Calculate factor 
    xf = tf.reshape(tf.matmul(X,fweight),[ss,k])
    # Calculate residual 
    xr = X - tf.matmul(xf,beta)
    # Concatenate factor and residual
    x_new = tf.concat([xf,xr],1)
    
    del xf,xr
    _ = gc.collect() #释放内存
       
    x_new = x_new[:,0:ncov+k]
    
    return x_new

'''
Logistic Regression
'''

IMSIZE = 784
input_layer = Input([784])
x = input_layer
x = Dense(10,activation = 'softmax')(x) 
output_layer=x
model=Model(input_layer,output_layer)

batch_size = 200
K1 = int(np.ceil(N0/batch_size))
K2 = int(np.ceil(N1/batch_size))
B = 200

time_cost1 = np.zeros(B)
time_checker = 0

train_loss1 = np.zeros(B)
test_loss1 = np.zeros(B)

train_acc1 = np.zeros(B)
test_acc1 = np.zeros(B)

optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
metrics = tf.keras.metrics.SparseCategoricalAccuracy()

for i in range(B):
    print("Current Epochs:",i+1)
    train_loss_avg = 0.
    test_loss_avg = 0.
    train_acc_avg = 0.
    test_acc_avg = 0.
    
    # Update a whole epoch
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
    time_cost1[i] = time_checker
    
    # Calculate training loss and acc
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
    
    # Calculate test loss and acc
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


# One factor 

fweight,beta,weight0 = Estimate_FactorX(X0[0:200,],1)
X0_new = Separate_FactorX(X0,fweight,beta)
X1_new = Separate_FactorX(X1,fweight,beta)
weight1 = tf.concat([weight0[0:1],weight0[1]*tf.ones(784)],axis = 0)
weight = tf.reshape((weight1[0]/weight1)**2,[785,1])


IMSIZE = 785
input_layer = Input([785])
x = input_layer   
x = Dense(10,activation = 'softmax')(x) 
output_layer=x
model1=Model(input_layer,output_layer)


batch_size = 200
K1 = int(np.ceil(N0/batch_size))
K2 = int(np.ceil(N1/batch_size))
B = 200

time_cost2 = np.zeros(B)
f_time_cost = np.zeros(B)
time_checker = 0

train_loss2 = np.zeros(B)
test_loss2 = np.zeros(B)

train_acc2 = np.zeros(B)
test_acc2= np.zeros(B)

optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
metrics = tf.keras.metrics.SparseCategoricalAccuracy()


for i in range(B):
    print("Current Epochs:",i+1)
    train_loss_avg = 0.
    test_loss_avg = 0.
    train_acc_avg = 0.
    test_acc_avg = 0.
    
    time_checker2 = 0
    
    
    start = time.time()
    for k in range(K1):
        
        X,Y = X0_new[k*batch_size:(k+1)*batch_size,],Y0[k*batch_size:(k+1)*batch_size].reshape(batch_size,) 
        with tf.GradientTape() as tape:
            Y_pred = model1(X)
            loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=Y, y_pred=Y_pred)
            loss = tf.reduce_mean(loss)
        grads = tape.gradient(loss, model1.variables)
        grads[0] = weight * grads[0]
        optimizer.apply_gradients(grads_and_vars=zip(grads, model1.variables))
    end = time.time()
    time_checker = time_checker + end-start
    time_cost2[i] = time_checker
    
    for k in range(K1):
        X,Y = X0_new[k*batch_size:(k+1)*batch_size,],Y0[k*batch_size:(k+1)*batch_size].reshape(batch_size,)
        Y_pred = model1(X)  
        loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=Y, y_pred=Y_pred)   
        loss = tf.reduce_mean(loss)                                                
        train_loss_avg = train_loss_avg + loss.numpy()
        metrics.update_state(y_true=Y, y_pred=Y_pred)
        train_acc_avg = train_acc_avg + metrics.result().numpy()    
    
    train_loss2[i] = train_loss_avg/K1
    train_acc2[i] = train_acc_avg/K1
    
    for k in range(K2):
        X,Y = X1_new[k*batch_size:(k+1)*batch_size,],Y1[k*batch_size:(k+1)*batch_size].reshape(batch_size,)
        Y_pred = model1(X)                    
        loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=Y, y_pred=Y_pred)   
        loss = tf.reduce_mean(loss)                                                
        test_loss_avg = test_loss_avg + loss.numpy()
        metrics.update_state(y_true=Y, y_pred=Y_pred)
        test_acc_avg = test_acc_avg + metrics.result().numpy()    
    
    test_loss2[i] = test_loss_avg/K2
    test_acc2[i] = test_acc_avg/K2


# 2 factors 


fweight,beta,weight0 = Estimate_FactorX(X0[0:200,],2)
X0_new = Separate_FactorX(X0,fweight,beta)
X1_new = Separate_FactorX(X1,fweight,beta)
weight1 = tf.concat([weight0[0:2],weight0[2]*tf.ones(784)],axis = 0)
weight = tf.reshape((weight1[0]/weight1)**2,[786,1])


input_layer = Input([786])
x = input_layer   
x = Dense(10,activation = 'softmax')(x) 
output_layer=x
model1=Model(input_layer,output_layer)


batch_size = 200
K1 = int(np.ceil(N0/batch_size))
K2 = int(np.ceil(N1/batch_size))
B = 200

time_cost3 = np.zeros(B)
f_time_cost = np.zeros(B)
time_checker = 0

train_loss3 = np.zeros(B)
test_loss3 = np.zeros(B)

train_acc3 = np.zeros(B)
test_acc3= np.zeros(B)

optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
metrics = tf.keras.metrics.SparseCategoricalAccuracy()


for i in range(B):
    print("Current Epochs:",i+1)
    train_loss_avg = 0.
    test_loss_avg = 0.
    train_acc_avg = 0.
    test_acc_avg = 0.
    
    time_checker2 = 0
    
    start = time.time()
    for k in range(K1):
        
        X,Y = X0_new[k*batch_size:(k+1)*batch_size,],Y0[k*batch_size:(k+1)*batch_size].reshape(batch_size,) 
        with tf.GradientTape() as tape:
            Y_pred = model1(X)
            loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=Y, y_pred=Y_pred)
            loss = tf.reduce_mean(loss)
        grads = tape.gradient(loss, model1.variables)
        grads[0] = weight * grads[0];
        optimizer.apply_gradients(grads_and_vars=zip(grads, model1.variables))
    end = time.time()
    time_checker = time_checker + end-start
    time_cost3[i] = time_checker
    
    for k in range(K1):
        X,Y = X0_new[k*batch_size:(k+1)*batch_size,],Y0[k*batch_size:(k+1)*batch_size].reshape(batch_size,)
        Y_pred = model1(X)  
        loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=Y, y_pred=Y_pred)   
        loss = tf.reduce_mean(loss)                                                
        train_loss_avg = train_loss_avg + loss.numpy()
        metrics.update_state(y_true=Y, y_pred=Y_pred)
        train_acc_avg = train_acc_avg + metrics.result().numpy()    
    
    train_loss3[i] = train_loss_avg/K1
    train_acc3[i] = train_acc_avg/K1
    
    for k in range(K2):
        X,Y = X1_new[k*batch_size:(k+1)*batch_size,],Y1[k*batch_size:(k+1)*batch_size].reshape(batch_size,)
        Y_pred = model1(X)                    
        loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=Y, y_pred=Y_pred)   
        loss = tf.reduce_mean(loss)                                                
        test_loss_avg = test_loss_avg + loss.numpy()
        metrics.update_state(y_true=Y, y_pred=Y_pred)
        test_acc_avg = test_acc_avg + metrics.result().numpy()    
    
    test_loss3[i] = test_loss_avg/K2
    test_acc3[i] = test_acc_avg/K2

    
# 3 factors 

fweight,beta,weight0 = Estimate_FactorX(X0[0:200,],3)
X0_new = Separate_FactorX(X0,fweight,beta)
X1_new = Separate_FactorX(X1,fweight,beta)
weight1 = tf.concat([weight0[0:3],weight0[3]*tf.ones(784)],axis = 0)
weight = tf.reshape((weight1[0]/weight1)**2,[787,1])


input_layer = Input([787])
x = input_layer   
x = Dense(10,activation = 'softmax')(x) 
output_layer=x
model1=Model(input_layer,output_layer)


batch_size = 200
K1 = int(np.ceil(N0/batch_size))
K2 = int(np.ceil(N1/batch_size))
B = 200

time_cost4 = np.zeros(B)
f_time_cost = np.zeros(B)
time_checker = 0

train_loss4 = np.zeros(B)
test_loss4 = np.zeros(B)

train_acc4 = np.zeros(B)
test_acc4= np.zeros(B)

optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
metrics = tf.keras.metrics.SparseCategoricalAccuracy()


for i in range(B):
    print("Current Epochs:",i+1)
    train_loss_avg = 0.
    test_loss_avg = 0.
    train_acc_avg = 0.
    test_acc_avg = 0.
    
    time_checker2 = 0
    
    start = time.time()
    for k in range(K1):
        
        X,Y = X0_new[k*batch_size:(k+1)*batch_size,],Y0[k*batch_size:(k+1)*batch_size].reshape(batch_size,) 
        with tf.GradientTape() as tape:
            Y_pred = model1(X)
            loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=Y, y_pred=Y_pred)
            loss = tf.reduce_mean(loss)
        grads = tape.gradient(loss, model1.variables)
        grads[0] = weight * grads[0]
        optimizer.apply_gradients(grads_and_vars=zip(grads, model1.variables))
    end = time.time()
    time_checker = time_checker + end-start
    time_cost4[i] = time_checker
    
    for k in range(K1):
        X,Y = X0_new[k*batch_size:(k+1)*batch_size,],Y0[k*batch_size:(k+1)*batch_size].reshape(batch_size,)
        Y_pred = model1(X)  
        loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=Y, y_pred=Y_pred)   
        loss = tf.reduce_mean(loss)                                                
        train_loss_avg = train_loss_avg + loss.numpy()
        metrics.update_state(y_true=Y, y_pred=Y_pred)
        train_acc_avg = train_acc_avg + metrics.result().numpy()    
    
    train_loss4[i] = train_loss_avg/K1
    train_acc4[i] = train_acc_avg/K1
    
    for k in range(K2):
        X,Y = X1_new[k*batch_size:(k+1)*batch_size,],Y1[k*batch_size:(k+1)*batch_size].reshape(batch_size,)
        Y_pred = model1(X)                    
        loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=Y, y_pred=Y_pred)   
        loss = tf.reduce_mean(loss)                                                
        test_loss_avg = test_loss_avg + loss.numpy()
        metrics.update_state(y_true=Y, y_pred=Y_pred)
        test_acc_avg = test_acc_avg + metrics.result().numpy()    
    
    test_loss4[i] = test_loss_avg/K2
    test_acc4[i] = test_acc_avg/K2

# 5 factors
fweight,beta,weight0 = Estimate_FactorX(X0[0:200,],5)
X0_new = Separate_FactorX(X0,fweight,beta)
X1_new = Separate_FactorX(X1,fweight,beta)
weight1 = tf.concat([weight0[0:5],weight0[5]*tf.ones(784)],axis = 0)
weight = tf.reshape((weight1[0]/weight1)**2,[789,1])


input_layer = Input([789])
x = input_layer   
x = Dense(10,activation = 'softmax')(x) 
output_layer=x
model1=Model(input_layer,output_layer)


batch_size = 200
K1 = int(np.ceil(N0/batch_size))
K2 = int(np.ceil(N1/batch_size))
B = 200

time_cost5 = np.zeros(B)
f_time_cost = np.zeros(B)
time_checker = 0

train_loss5 = np.zeros(B)
test_loss5 = np.zeros(B)

train_acc5 = np.zeros(B)
test_acc5= np.zeros(B)

optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
metrics = tf.keras.metrics.SparseCategoricalAccuracy()


for i in range(B):
    print("Current Epochs:",i+1)
    train_loss_avg = 0.
    test_loss_avg = 0.
    train_acc_avg = 0.
    test_acc_avg = 0.
    
    time_checker2 = 0
    
    start = time.time()
    for k in range(K1):
        
        X,Y = X0_new[k*batch_size:(k+1)*batch_size,],Y0[k*batch_size:(k+1)*batch_size].reshape(batch_size,) 
        with tf.GradientTape() as tape:
            Y_pred = model1(X)
            loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=Y, y_pred=Y_pred)
            loss = tf.reduce_mean(loss)
        grads = tape.gradient(loss, model1.variables)
        grads[0] = weight * grads[0]
        optimizer.apply_gradients(grads_and_vars=zip(grads, model1.variables))
    end = time.time()
    time_checker = time_checker + end-start
    time_cost5[i] = time_checker
    
    for k in range(K1):
        X,Y = X0_new[k*batch_size:(k+1)*batch_size,],Y0[k*batch_size:(k+1)*batch_size].reshape(batch_size,)
        Y_pred = model1(X)  
        loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=Y, y_pred=Y_pred)   
        loss = tf.reduce_mean(loss)                                                
        train_loss_avg = train_loss_avg + loss.numpy()
        metrics.update_state(y_true=Y, y_pred=Y_pred)
        train_acc_avg = train_acc_avg + metrics.result().numpy()    
    
    train_loss5[i] = train_loss_avg/K1
    train_acc5[i] = train_acc_avg/K1
    
    for k in range(K2):
        X,Y = X1_new[k*batch_size:(k+1)*batch_size,],Y1[k*batch_size:(k+1)*batch_size].reshape(batch_size,)
        Y_pred = model1(X)                    
        loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=Y, y_pred=Y_pred)   
        loss = tf.reduce_mean(loss)                                               
        test_loss_avg = test_loss_avg + loss.numpy()
        metrics.update_state(y_true=Y, y_pred=Y_pred)
        test_acc_avg = test_acc_avg + metrics.result().numpy()    
    
    test_loss5[i] = test_loss_avg/K2
    test_acc5[i] = test_acc_avg/K2


# 10 factors

fweight,beta,weight0 = Estimate_FactorX(X0[0:200,],10)
X0_new = Separate_FactorX(X0,fweight,beta)
X1_new = Separate_FactorX(X1,fweight,beta)
weight1 = tf.concat([weight0[0:10],weight0[10]*tf.ones(784)],axis = 0)
weight = tf.reshape((weight1[0]/weight1)**2,[794,1])


input_layer = Input([794])
x = input_layer   
x = Dense(10,activation = 'softmax')(x) 
output_layer=x
model1=Model(input_layer,output_layer)


batch_size = 200
K1 = int(np.ceil(N0/batch_size))
K2 = int(np.ceil(N1/batch_size))
B = 200

time_cost6 = np.zeros(B)
f_time_cost = np.zeros(B)
time_checker = 0

train_loss6 = np.zeros(B)
test_loss6 = np.zeros(B)

train_acc6 = np.zeros(B)
test_acc6= np.zeros(B)

optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
metrics = tf.keras.metrics.SparseCategoricalAccuracy()


for i in range(B):
    print("Current Epochs:",i+1)
    train_loss_avg = 0.
    test_loss_avg = 0.
    train_acc_avg = 0.
    test_acc_avg = 0.
    
    time_checker2 = 0
    
    start = time.time()
    for k in range(K1):
        
        X,Y = X0_new[k*batch_size:(k+1)*batch_size,],Y0[k*batch_size:(k+1)*batch_size].reshape(batch_size,) 
        with tf.GradientTape() as tape:
            Y_pred = model1(X)
            loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=Y, y_pred=Y_pred)
            loss = tf.reduce_mean(loss)
        grads = tape.gradient(loss, model1.variables)
        grads[0] = weight * grads[0]
        optimizer.apply_gradients(grads_and_vars=zip(grads, model1.variables))
    end = time.time()
    time_checker = time_checker + end-start
    time_cost6[i] = time_checker
    
    for k in range(K1):
        X,Y = X0_new[k*batch_size:(k+1)*batch_size,],Y0[k*batch_size:(k+1)*batch_size].reshape(batch_size,)
        Y_pred = model1(X)  
        loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=Y, y_pred=Y_pred)   
        loss = tf.reduce_mean(loss)                                                
        train_loss_avg = train_loss_avg + loss.numpy()
        metrics.update_state(y_true=Y, y_pred=Y_pred)
        train_acc_avg = train_acc_avg + metrics.result().numpy()    
    
    train_loss6[i] = train_loss_avg/K1
    train_acc6[i] = train_acc_avg/K1
    
    for k in range(K2):
        X,Y = X1_new[k*batch_size:(k+1)*batch_size,],Y1[k*batch_size:(k+1)*batch_size].reshape(batch_size,)
        Y_pred = model1(X)                    
        loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=Y, y_pred=Y_pred)   
        loss = tf.reduce_mean(loss)                                                
        test_loss_avg = test_loss_avg + loss.numpy()
        metrics.update_state(y_true=Y, y_pred=Y_pred)
        test_acc_avg = test_acc_avg + metrics.result().numpy()    
    
    test_loss6[i] = test_loss_avg/K2
    test_acc6[i] = test_acc_avg/K2


# 50 factors

fweight,beta,weight0 = Estimate_FactorX(X0[0:200,],50)
X0_new = Separate_FactorX(X0,fweight,beta)
X1_new = Separate_FactorX(X1,fweight,beta)
weight1 = tf.concat([weight0[0:50],weight0[50]*tf.ones(784)],axis = 0)
weight = tf.reshape((weight1[0]/weight1)**2,[834,1])

input_layer = Input([834])
x = input_layer   
x = Dense(10,activation = 'softmax')(x) 
output_layer=x
model1=Model(input_layer,output_layer)

batch_size = 200
K1 = int(np.ceil(N0/batch_size))
K2 = int(np.ceil(N1/batch_size))
B = 200

time_cost7 = np.zeros(B)
f_time_cost = np.zeros(B)
time_checker = 0

train_loss7 = np.zeros(B)
test_loss7 = np.zeros(B)

train_acc7 = np.zeros(B)
test_acc7= np.zeros(B)

optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
metrics = tf.keras.metrics.SparseCategoricalAccuracy()


for i in range(B):
    print("Current Epochs:",i+1)
    train_loss_avg = 0.
    test_loss_avg = 0.
    train_acc_avg = 0.
    test_acc_avg = 0.
    
    time_checker2 = 0
    
    start = time.time()
    for k in range(K1):
        
        X,Y = X0_new[k*batch_size:(k+1)*batch_size,],Y0[k*batch_size:(k+1)*batch_size].reshape(batch_size,) 
        with tf.GradientTape() as tape:
            Y_pred = model1(X)
            loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=Y, y_pred=Y_pred)
            loss = tf.reduce_mean(loss)
        grads = tape.gradient(loss, model1.variables)
        grads[0] = weight * grads[0]
        optimizer.apply_gradients(grads_and_vars=zip(grads, model1.variables))
    end = time.time()
    time_checker = time_checker + end-start
    time_cost7[i] = time_checker
    
    for k in range(K1):
        X,Y = X0_new[k*batch_size:(k+1)*batch_size,],Y0[k*batch_size:(k+1)*batch_size].reshape(batch_size,)
        Y_pred = model1(X)  
        loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=Y, y_pred=Y_pred)   
        loss = tf.reduce_mean(loss)                                                
        train_loss_avg = train_loss_avg + loss.numpy()
        metrics.update_state(y_true=Y, y_pred=Y_pred)
        train_acc_avg = train_acc_avg + metrics.result().numpy()    
    
    train_loss7[i] = train_loss_avg/K1
    train_acc7[i] = train_acc_avg/K1
    
    for k in range(K2):
        X,Y = X1_new[k*batch_size:(k+1)*batch_size,],Y1[k*batch_size:(k+1)*batch_size].reshape(batch_size,)
        Y_pred = model1(X)                    
        loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=Y, y_pred=Y_pred)   
        loss = tf.reduce_mean(loss)                                                
        test_loss_avg = test_loss_avg + loss.numpy()
        metrics.update_state(y_true=Y, y_pred=Y_pred)
        test_acc_avg = test_acc_avg + metrics.result().numpy()    
    
    test_loss7[i] = test_loss_avg/K2
    test_acc7[i] = test_acc_avg/K2


# 100 factors


fweight,beta,weight0 = Estimate_FactorX(X0[0:200,],100)
X0_new = Separate_FactorX(X0,fweight,beta)
X1_new = Separate_FactorX(X1,fweight,beta)
weight1 = tf.concat([weight0[0:100],weight0[100]*tf.ones(784)],axis = 0)
weight = tf.reshape((weight1[0]/weight1)**2,[884,1])

input_layer = Input([884])
x = input_layer   
x = Dense(10,activation = 'softmax')(x) 
output_layer=x
model1=Model(input_layer,output_layer)


batch_size = 200
K1 = int(np.ceil(N0/batch_size))
K2 = int(np.ceil(N1/batch_size))
B = 200

time_cost8 = np.zeros(B)
f_time_cost = np.zeros(B)
time_checker = 0

train_loss8 = np.zeros(B)
test_loss8 = np.zeros(B)

train_acc8 = np.zeros(B)
test_acc8 = np.zeros(B)

optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
metrics = tf.keras.metrics.SparseCategoricalAccuracy()


for i in range(B):
    print("Current Epochs:",i+1)
    train_loss_avg = 0.
    test_loss_avg = 0.
    train_acc_avg = 0.
    test_acc_avg = 0.
    
    time_checker2 = 0
    
    start = time.time()
    for k in range(K1):
        
        X,Y = X0_new[k*batch_size:(k+1)*batch_size,],Y0[k*batch_size:(k+1)*batch_size].reshape(batch_size,) 
        with tf.GradientTape() as tape:
            Y_pred = model1(X)
            loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=Y, y_pred=Y_pred)
            loss = tf.reduce_mean(loss)
        grads = tape.gradient(loss, model1.variables)
        grads[0] = weight * grads[0]
        optimizer.apply_gradients(grads_and_vars=zip(grads, model1.variables))
    end = time.time()
    time_checker = time_checker + end-start
    time_cost8[i] = time_checker
    
    for k in range(K1):
        X,Y = X0_new[k*batch_size:(k+1)*batch_size,],Y0[k*batch_size:(k+1)*batch_size].reshape(batch_size,)
        Y_pred = model1(X)  
        loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=Y, y_pred=Y_pred)   
        loss = tf.reduce_mean(loss)                                                
        train_loss_avg = train_loss_avg + loss.numpy()
        metrics.update_state(y_true=Y, y_pred=Y_pred)
        train_acc_avg = train_acc_avg + metrics.result().numpy()    
    
    train_loss8[i] = train_loss_avg/K1
    train_acc8[i] = train_acc_avg/K1
    
    for k in range(K2):
        X,Y = X1_new[k*batch_size:(k+1)*batch_size,],Y1[k*batch_size:(k+1)*batch_size].reshape(batch_size,)
        Y_pred = model1(X)                    
        loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=Y, y_pred=Y_pred)   
        loss = tf.reduce_mean(loss)                                                
        test_loss_avg = test_loss_avg + loss.numpy()
        metrics.update_state(y_true=Y, y_pred=Y_pred)
        test_acc_avg = test_acc_avg + metrics.result().numpy()    
    
    test_loss8[i] = test_loss_avg/K2
    test_acc8[i] = test_acc_avg/K2



df = pd.DataFrame([time_cost1,train_loss1,train_acc1,test_loss1,test_acc1,
                   time_cost2,train_loss2,train_acc2,test_loss2,test_acc2,
                   time_cost3,train_loss3,train_acc3,test_loss3,test_acc3,
                   time_cost4,train_loss4,train_acc4,test_loss4,test_acc4,
                   time_cost5,train_loss5,train_acc5,test_loss5,test_acc5,
                   time_cost6,train_loss6,train_acc6,test_loss6,test_acc6,
                   time_cost7,train_loss7,train_acc7,test_loss7,test_acc7,
                   time_cost8,train_loss8,train_acc8,test_loss8,test_acc8])
new_col = ["time_cost1","train_loss1","train_acc1","test_loss1","test_acc1",
           "time_cost2","train_loss2","train_acc2","test_loss2","test_acc2",
           "time_cost3","train_loss3","train_acc3","test_loss3","test_acc3",
           "time_cost4","train_loss4","train_acc4","test_loss4","test_acc4",
           "time_cost5","train_loss5","train_acc5","test_loss5","test_acc5",
           "time_cost6","train_loss6","train_acc6","test_loss6","test_acc6",
           "time_cost7","train_loss7","train_acc7","test_loss7","test_acc7",
           "time_cost8","train_loss8","train_acc8","test_loss8","test_acc8"]


df2 = pd.DataFrame(df.values.T, columns=new_col)


df2.to_csv('./MNIST_Logistic.csv', index=False, header=True)


from matplotlib.backends.backend_pdf import PdfPages
pdf = PdfPages('./MNIST.pdf')

fig, ax = plt.subplots(1,2) 
fig.set_figheight(5)
fig.set_figwidth(13)
ax[0].grid(color='grey',
        linestyle='--',
        linewidth=1,
        alpha=0.3)
ax[0].set_xlabel('Time(second)')
ax[0].set_ylabel('Training Loss')
ax[0].set_title('MNIST Logistic Regression (batchsize = 200)')
ax[0].plot(time_cost1, train_loss1,linewidth=0.8) 
ax[0].plot(time_cost2, train_loss2,linewidth=0.8) 
#ax[0].plot(time_cost3, train_loss3,linewidth=0.8) 
#ax[0].plot(time_cost5, train_loss4,linewidth=0.8)
ax[0].plot(time_cost6, train_loss4,linewidth=0.8)
ax[0].plot(time_cost7, train_loss5,linewidth=0.8)




ax[1].grid(color='grey',
        linestyle='--',
        linewidth=1,
        alpha=0.3)
ax[1].set_xlabel('Time(second)')
ax[1].set_ylabel('Accuracy')
ax[1].set_title('MNIST Logistic Regression (batchsize = 200)')
ax[1].plot(time_cost1, test_acc1,linewidth=0.8) 
ax[1].plot(time_cost2, test_acc2,linewidth=0.8) 
ax[1].plot(time_cost6, test_acc4,linewidth=0.8) 
ax[1].plot(time_cost7, test_acc5,linewidth=0.8) 

ax[0].legend(["SGD","FN (d=1)","FN (d=2)","FN (d=10)","FN (d=50)"] , loc = "best")  
ax[1].legend(["SGD","FN (d=1)","FN (d=2)","FN (d=10)","FN (d=50)"] , loc = "best")     


pdf.savefig()
plt.show()

plt.close()
pdf.close()


# In[ ]:


ax[1][0].grid(color='grey',
        linestyle='--',
        linewidth=1,
        alpha=0.3)
ax[1][0].set_xlabel('Time(second)')
ax[1][0].set_ylabel('log-Loss')
ax[1][0].set_title('MNIST Test Data (batchsize = 200)')
ax[1][0].plot(time_cost1, np.log(test_loss1),linewidth=0.8) 
ax[1][0].plot(time_cost2, np.log(test_loss2),linewidth=0.8) 
ax[1][0].plot(time_cost3, np.log(test_loss3),linewidth=0.8) 
ax[1][0].plot(time_cost5, np.log(test_loss4),linewidth=0.8)
ax[1][0].plot(time_cost6, np.log(test_loss4),linewidth=0.8)
ax[1][0].plot(time_cost7, np.log(test_loss5),linewidth=0.8)


ax[0][1].grid(color='grey',
        linestyle='--',
        linewidth=1,
        alpha=0.3)
ax[0][1].set_xlabel('Time(second)')
ax[0][1].set_ylabel('Accuracy')
ax[0][1].set_title('MNIST Logistic Regression (batchsize = 200)')
ax[0][1].plot(time_cost1, train_acc1,linewidth=0.8) 
ax[0][1].plot(time_cost2, train_acc2,linewidth=0.8) 
ax[0][1].plot(time_cost3, train_acc3,linewidth=0.8) 
ax[0][1].plot(time_cost5, train_acc4,linewidth=0.8)
ax[0][1].plot(time_cost6, train_acc4,linewidth=0.8)
ax[0][1].plot(time_cost7, train_acc5,linewidth=0.8)


