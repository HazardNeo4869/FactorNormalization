'''
Experiment: MNIST + Multilayer NNs
'''

# 载入必要的库
import numpy as np
import tensorflow as tf    #tf.compat.v1
import gc
import time
import pandas as pd
from keras.layers import Dense,Flatten, Input, concatenate
from keras import Model
from matplotlib import pyplot as plt


def fn_model(old_model,num_class):
    
    num_layers =  len(old_model.layers) 
    print("Old model has " + str(num_layers) + " layers in total! ")
    model0 = Model(inputs=old_model.input, outputs=old_model.layers[num_layers-2].output)
    
    input_z = Input([1])
    output_z = input_z
    model1 = Model(input_z, output_z)
    
    combined = concatenate([model0.output, model1.output])
    z = Dense(num_class, activation="softmax")(combined) 
    new_model = Model(inputs=[model0.input, model1.input], outputs=z)
    
    return new_model

'''
Estimate_FactorX
'''
def Estimate_FactorX(X,k):
    
    # 转化为tf格式
    X = tf.constant(X, dtype = tf.float32)
    ss,ncov = X.shape
     
    # 添加偏置项
    X = tf.concat([X,tf.ones([ss,1])],1)
    beta = tf.zeros(1)
    
    
    # SVD 分解
    s,u,v = tf.compat.v1.svd(X,full_matrices=True)     # svd分解
    fv = v[:,0:k]
    
    del u,v
    _ = gc.collect() #释放内存
    
    # 计算最大因子
    xf = tf.reshape(tf.matmul(X,fv),[ss,k])
    # 计算残差回归系数
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
    # 添加偏置项
    X = tf.concat([X,tf.ones([ss,1])],1)
    # 计算因子
    xf = tf.reshape(tf.matmul(X,fweight),[ss,k])
    # 计算因子残差
    xr = X - tf.matmul(xf,beta)
    # 合并因子变量和因子残差
    x_new = tf.concat([xf,xr],1)
    
    del xf,xr
    _ = gc.collect() #释放内存
    
    x_new = x_new[:,0:ncov+k]
    
    return x_new


'''
STEP.1: Load Data and Factor Decomposition
'''
# 读取
mnist = tf.keras.datasets.mnist
(X0, Y0), (X1, Y1) = mnist.load_data()

# 整理格式
N0=X0.shape[0];N1=X1.shape[0]
X0 = X0.reshape(N0,784)/255.0
X1 = X1.reshape(N1,784)/255.0
Y0 = Y0.reshape(60000,1)
Y1 = Y1.reshape(10000,1)

# 数据标准化
mean = np.mean(X0,axis=(0,1))
std = np.std(X0,axis=(0,1))
X0 = (X0-mean)/(std+1e-7)
X1 = (X1-mean)/(std+1e-7)

X0 = X0.astype(np.float32)
X1 = X1.astype(np.float32)




'''
STEP.2: Model Reconstruction
'''

IMSIZE = 784
input_layer = Input([784])
x = input_layer
x = Dense(1000, kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=1), bias_initializer='zeros',activation = "relu")(x)
x = Dense(1000, kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=2), bias_initializer='zeros',activation = "relu")(x)
x = Dense(10, kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=3), bias_initializer='zeros',activation = "softmax")(x)
output_layer=x
model=Model(input_layer,output_layer)


'''
STEP.3: Training
'''

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

optimizer = tf.keras.optimizers.SGD(learning_rate=0.05)
metrics = tf.keras.metrics.SparseCategoricalAccuracy()

for i in range(B):
    print("Current Epochs:",i+1)
    train_loss_avg = 0.
    test_loss_avg = 0.
    train_acc_avg = 0.
    test_acc_avg = 0.
    

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



## Lr = 0.01

IMSIZE = 784
input_layer = Input([784])
x = input_layer
x = Dense(1000, kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=1), bias_initializer='zeros',activation = "relu")(x)
x = Dense(1000, kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=2), bias_initializer='zeros',activation = "relu")(x)
x = Dense(10, kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=3), bias_initializer='zeros',activation = "softmax")(x)
output_layer=x
model=Model(input_layer,output_layer)


batch_size = 200
K1 = int(np.ceil(N0/batch_size))
K2 = int(np.ceil(N1/batch_size))
B = 200

time_cost2 = np.zeros(B)
time_checker = 0

train_loss2 = np.zeros(B)
test_loss2 = np.zeros(B)

train_acc2 = np.zeros(B)
test_acc2 = np.zeros(B)

optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
metrics = tf.keras.metrics.SparseCategoricalAccuracy()

for i in range(B):
    print("Current Epochs:",i+1)
    train_loss_avg = 0.
    test_loss_avg = 0.
    train_acc_avg = 0.
    test_acc_avg = 0.
    
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
    time_cost2[i] = time_checker
    
    
    for k in range(K1):
        X,Y = X0[k*batch_size:(k+1)*batch_size,],Y0[k*batch_size:(k+1)*batch_size].reshape(batch_size,)
        Y_pred = model(X)                                                          
        loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=Y, y_pred=Y_pred)   
        loss = tf.reduce_mean(loss)                                                
        train_loss_avg = train_loss_avg + loss.numpy()
        metrics.update_state(y_true=Y, y_pred=Y_pred)
        train_acc_avg = train_acc_avg + metrics.result().numpy()    
    
    train_loss2[i] = train_loss_avg/K1
    train_acc2[i] = train_acc_avg/K1
    
    for k in range(K2):
        X,Y = X1[k*batch_size:(k+1)*batch_size,],Y1[k*batch_size:(k+1)*batch_size].reshape(batch_size,)
        Y_pred = model(X)                                                          
        loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=Y, y_pred=Y_pred)   
        loss = tf.reduce_mean(loss)                                                
        test_loss_avg = test_loss_avg + loss.numpy()
        metrics.update_state(y_true=Y, y_pred=Y_pred)
        test_acc_avg = test_acc_avg + metrics.result().numpy()    
    
    test_loss2[i] = test_loss_avg/K2
    test_acc2[i] = test_acc_avg/K2


## Lr = 0.005

IMSIZE = 784
input_layer = Input([784])
x = input_layer
x = Dense(1000, kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=1), bias_initializer='zeros',activation = "relu")(x)
x = Dense(1000, kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=2), bias_initializer='zeros',activation = "relu")(x)
x = Dense(10, kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=3), bias_initializer='zeros',activation = "softmax")(x)
output_layer=x
model=Model(input_layer,output_layer)


batch_size = 200
K1 = int(np.ceil(N0/batch_size))
K2 = int(np.ceil(N1/batch_size))
B = 200

time_cost3 = np.zeros(B)
time_checker = 0

train_loss3 = np.zeros(B)
test_loss3 = np.zeros(B)

train_acc3 = np.zeros(B)
test_acc3 = np.zeros(B)

optimizer = tf.keras.optimizers.SGD(learning_rate=0.005)
metrics = tf.keras.metrics.SparseCategoricalAccuracy()

for i in range(B):
    print("Current Epochs:",i+1)
    train_loss_avg = 0.
    test_loss_avg = 0.
    train_acc_avg = 0.
    test_acc_avg = 0.
    
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
    time_cost3[i] = time_checker
    
    for k in range(K1):
        X,Y = X0[k*batch_size:(k+1)*batch_size,],Y0[k*batch_size:(k+1)*batch_size].reshape(batch_size,)
        Y_pred = model(X)                                                          
        loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=Y, y_pred=Y_pred)   
        loss = tf.reduce_mean(loss)                                                
        train_loss_avg = train_loss_avg + loss.numpy()
        metrics.update_state(y_true=Y, y_pred=Y_pred)
        train_acc_avg = train_acc_avg + metrics.result().numpy()    
    
    train_loss3[i] = train_loss_avg/K1
    train_acc3[i] = train_acc_avg/K1
    
    for k in range(K2):
        X,Y = X1[k*batch_size:(k+1)*batch_size,],Y1[k*batch_size:(k+1)*batch_size].reshape(batch_size,)
        Y_pred = model(X)                                                          
        loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=Y, y_pred=Y_pred)   
        loss = tf.reduce_mean(loss)                                                
        test_loss_avg = test_loss_avg + loss.numpy()
        metrics.update_state(y_true=Y, y_pred=Y_pred)
        test_acc_avg = test_acc_avg + metrics.result().numpy()    
    
    test_loss3[i] = test_loss_avg/K2
    test_acc3[i] = test_acc_avg/K2


## FN 


fweight,beta,weight0 = Estimate_FactorX(X0[0:200,],1)
X0_new = Separate_FactorX(X0,fweight,beta)
X1_new = Separate_FactorX(X1,fweight,beta)
weight1 = tf.concat([weight0[1]*tf.ones(1000),weight0[0:1]],axis = 0)
weight = tf.reshape((weight0[0]/weight1)**2,[1001,1])

X00 = X0_new[:,1:3073];Z0 = X0_new[:,0:1]
X11 = X1_new[:,1:3073];Z1 = X1_new[:,0:1]


# Lr = 0.05


IMSIZE = 784
input_layer = Input([784])
x = input_layer
x = Dense(1000, kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=1), bias_initializer='zeros',activation = "relu")(x)
x = Dense(1000, kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=2), bias_initializer='zeros',activation = "relu")(x)
x = Dense(10, kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=3), bias_initializer='zeros',activation = "softmax")(x)
output_layer=x
model=Model(input_layer,output_layer)

model1 = fn_model(model,10)


batch_size = 200
K1 = int(np.ceil(N0/batch_size))
K2 = int(np.ceil(N1/batch_size))
B = 200  # epoch个数

time_cost4 = np.zeros(B)    
time_checker = 0          


train_loss4 = np.zeros(B)
test_loss4 = np.zeros(B)

train_acc4 = np.zeros(B)
test_acc4= np.zeros(B)

optimizer = tf.keras.optimizers.SGD(learning_rate=0.05)      
metrics = tf.keras.metrics.SparseCategoricalAccuracy()       

                                           
for i in range(B):
    print("Current Epochs:",i+1)
    train_loss_avg = 0.
    test_loss_avg = 0.
    train_acc_avg = 0.
    test_acc_avg = 0.
    

    start = time.time()
    for k in range(K1):
        
        X,Y = X00[k*batch_size:(k+1)*batch_size,],Y0[k*batch_size:(k+1)*batch_size].reshape(batch_size,)
        Z = Z0[k*batch_size:(k+1)*batch_size,]
        
        with tf.GradientTape() as tape:
            Y_pred = model1([X,Z])
            loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=Y, y_pred=Y_pred)
            loss = tf.reduce_mean(loss)
        grads = tape.gradient(loss, model1.variables)
        l = len(grads)
        
        grads[l-2] = weight * grads[l-2]
        optimizer.apply_gradients(grads_and_vars=zip(grads, model1.variables))
    end = time.time()
    time_checker = time_checker + end-start
    time_cost4[i] = time_checker
    
    
    for k in range(K1):
        X,Y = X00[k*batch_size:(k+1)*batch_size,],Y0[k*batch_size:(k+1)*batch_size].reshape(batch_size,)
        Z = Z0[k*batch_size:(k+1)*batch_size,]
        Y_pred = model1([X,Z])                                             
        loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=Y, y_pred=Y_pred)       
        loss = tf.reduce_mean(loss)                                         
        train_loss_avg = train_loss_avg + loss.numpy()
        metrics.update_state(y_true=Y, y_pred=Y_pred)
        train_acc_avg = train_acc_avg + metrics.result().numpy()    
    
    train_loss4[i] = train_loss_avg/K1
    train_acc4[i] = train_acc_avg/K1
    
    for k in range(K2):
        X,Y = X11[k*batch_size:(k+1)*batch_size,],Y1[k*batch_size:(k+1)*batch_size].reshape(batch_size,)
        Z = Z1[k*batch_size:(k+1)*batch_size,]
        Y_pred = model1([X,Z])                                                          
        loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=Y, y_pred=Y_pred)   
        loss = tf.reduce_mean(loss)                                                
        test_loss_avg = test_loss_avg + loss.numpy()
        metrics.update_state(y_true=Y, y_pred=Y_pred)
        test_acc_avg = test_acc_avg + metrics.result().numpy()    
    
    test_loss4[i] = test_loss_avg/K2
    test_acc4[i] = test_acc_avg/K2



IMSIZE = 784
input_layer = Input([784])
x = input_layer
x = Dense(1000, kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=1), bias_initializer='zeros',activation = "relu")(x)
x = Dense(1000, kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=2), bias_initializer='zeros',activation = "relu")(x)
x = Dense(10, kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=3), bias_initializer='zeros',activation = "softmax")(x)
output_layer=x
model=Model(input_layer,output_layer)

model1 = fn_model(model,10)


batch_size = 200
K1 = int(np.ceil(N0/batch_size))
K2 = int(np.ceil(N1/batch_size))
B = 200  

time_cost5 = np.zeros(B)    
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
        
        X,Y = X00[k*batch_size:(k+1)*batch_size,],Y0[k*batch_size:(k+1)*batch_size].reshape(batch_size,)
        Z = Z0[k*batch_size:(k+1)*batch_size,]
        
        with tf.GradientTape() as tape:
            Y_pred = model1([X,Z])
            loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=Y, y_pred=Y_pred)
            loss = tf.reduce_mean(loss)
        grads = tape.gradient(loss, model1.variables)
        l = len(grads)
        
        grads[l-2] = weight * grads[l-2]
        optimizer.apply_gradients(grads_and_vars=zip(grads, model1.variables))
    end = time.time()
    time_checker = time_checker + end-start
    time_cost5[i] = time_checker
    
    
    for k in range(K1):
        X,Y = X00[k*batch_size:(k+1)*batch_size,],Y0[k*batch_size:(k+1)*batch_size].reshape(batch_size,)
        Z = Z0[k*batch_size:(k+1)*batch_size,]
        Y_pred = model1([X,Z])                                             
        loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=Y, y_pred=Y_pred)       
        loss = tf.reduce_mean(loss)                                         
        train_loss_avg = train_loss_avg + loss.numpy()
        metrics.update_state(y_true=Y, y_pred=Y_pred)
        train_acc_avg = train_acc_avg + metrics.result().numpy()    
    
    train_loss5[i] = train_loss_avg/K1
    train_acc5[i] = train_acc_avg/K1
    
    for k in range(K2):
        X,Y = X11[k*batch_size:(k+1)*batch_size,],Y1[k*batch_size:(k+1)*batch_size].reshape(batch_size,)
        Z = Z1[k*batch_size:(k+1)*batch_size,]
        Y_pred = model1([X,Z])                                                          
        loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=Y, y_pred=Y_pred)   
        loss = tf.reduce_mean(loss)                                                
        test_loss_avg = test_loss_avg + loss.numpy()
        metrics.update_state(y_true=Y, y_pred=Y_pred)
        test_acc_avg = test_acc_avg + metrics.result().numpy()    
    
    test_loss5[i] = test_loss_avg/K2
    test_acc5[i] = test_acc_avg/K2




IMSIZE = 784
input_layer = Input([784])
x = input_layer
x = Dense(1000, kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=1), bias_initializer='zeros',activation = "relu")(x)
x = Dense(1000, kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=2), bias_initializer='zeros',activation = "relu")(x)
x = Dense(10, kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=3), bias_initializer='zeros',activation = "softmax")(x)
output_layer=x
model=Model(input_layer,output_layer)

model1 = fn_model(model,10)


batch_size = 200
K1 = int(np.ceil(N0/batch_size))
K2 = int(np.ceil(N1/batch_size))
B = 200  

time_cost6 = np.zeros(B)    
time_checker = 0          


train_loss6 = np.zeros(B)
test_loss6 = np.zeros(B)

train_acc6 = np.zeros(B)
test_acc6= np.zeros(B)

optimizer = tf.keras.optimizers.SGD(learning_rate=0.005)      
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
        
        X,Y = X00[k*batch_size:(k+1)*batch_size,],Y0[k*batch_size:(k+1)*batch_size].reshape(batch_size,)
        Z = Z0[k*batch_size:(k+1)*batch_size,]
        
        with tf.GradientTape() as tape:
            Y_pred = model1([X,Z])
            loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=Y, y_pred=Y_pred)
            loss = tf.reduce_mean(loss)
        grads = tape.gradient(loss, model1.variables)
        l = len(grads)
        grads[l-2] = weight * grads[l-2]
        optimizer.apply_gradients(grads_and_vars=zip(grads, model1.variables))
    end = time.time()
    time_checker = time_checker + end-start
    time_cost6[i] = time_checker
    
    for k in range(K1):
        X,Y = X00[k*batch_size:(k+1)*batch_size,],Y0[k*batch_size:(k+1)*batch_size].reshape(batch_size,)
        Z = Z0[k*batch_size:(k+1)*batch_size,]
        Y_pred = model1([X,Z])                                             
        loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=Y, y_pred=Y_pred)       
        loss = tf.reduce_mean(loss)                                                                 train_loss_avg = train_loss_avg + loss.numpy()
        metrics.update_state(y_true=Y, y_pred=Y_pred)
        train_acc_avg = train_acc_avg + metrics.result().numpy()    
    
    train_loss6[i] = train_loss_avg/K1
    train_acc6[i] = train_acc_avg/K1
    
    for k in range(K2):
        X,Y = X11[k*batch_size:(k+1)*batch_size,],Y1[k*batch_size:(k+1)*batch_size].reshape(batch_size,)
        Z = Z1[k*batch_size:(k+1)*batch_size,]
        #Z  = (Z-np.mean(Z))/(np.std(Z)+1e-6)
        Y_pred = model1([X,Z])                                                          
        loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=Y, y_pred=Y_pred)   
        loss = tf.reduce_mean(loss)                                                
        test_loss_avg = test_loss_avg + loss.numpy()
        metrics.update_state(y_true=Y, y_pred=Y_pred)
        test_acc_avg = test_acc_avg + metrics.result().numpy()    
    
    test_loss6[i] = test_loss_avg/K2
    test_acc6[i] = test_acc_avg/K2


df = pd.DataFrame([time_cost1,train_loss1,train_acc1,test_loss1,test_acc1,
                   time_cost2,train_loss2,train_acc2,test_loss2,test_acc2,
                   time_cost3,train_loss3,train_acc3,test_loss3,test_acc3,
                   time_cost4,train_loss4,train_acc4,test_loss4,test_acc4,
                   time_cost5,train_loss5,train_acc5,test_loss5,test_acc5,
                   time_cost6,train_loss6,train_acc6,test_loss6,test_acc6])
new_col = ["time_cost1","train_loss1","train_acc1","test_loss1","test_acc1",
           "time_cost2","train_loss2","train_acc2","test_loss2","test_acc2",
           "time_cost3","train_loss3","train_acc3","test_loss3","test_acc3",
           "time_cost4","train_loss4","train_acc4","test_loss4","test_acc4",
           "time_cost5","train_loss5","train_acc5","test_loss5","test_acc5",
           "time_cost6","train_loss6","train_acc6","test_loss6","test_acc6"]

df2 = pd.DataFrame(df.values.T, columns=new_col)
df2.to_csv('./MNIST_MLP.csv', index=False, header=True)

from matplotlib.backends.backend_pdf import PdfPages

 
pdf = PdfPages('./MNIST_MLP.pdf')

fig, ax = plt.subplots(1,2) 
fig.set_figheight(7.5)
fig.set_figwidth(17)
ax[0].grid(color='grey',
        linestyle='--',
        linewidth=1,
        alpha=0.3)
ax[0].set_xlabel('Time(second)')
ax[0].set_ylabel('Training Loss')
ax[0].set_title('MNIST Multi-layers Neural Networks (batchsize = 200)')
ax[0].plot(time_cost1, np.log(train_loss1),linewidth=0.8) 
ax[0].plot(time_cost4, np.log(train_loss4),linewidth=0.8)
ax[0].plot(time_cost2, np.log(train_loss2),linewidth=0.8) 
ax[0].plot(time_cost5, np.log(train_loss5),linewidth=0.8) 
ax[0].plot(time_cost3, np.log(train_loss3),linewidth=0.8) 
ax[0].plot(time_cost6, np.log(train_loss6),linewidth=0.8)


ax[1].grid(color='grey',
        linestyle='--',
        linewidth=1,
        alpha=0.3)
ax[1].set_xlabel('Time(second)')
ax[1].set_ylabel('Accuracy')
ax[1].set_title('MNIST Multi-layers Neural Networks(batchsize = 200)')
ax[1].plot(time_cost1, test_acc1,linewidth=0.8) 
ax[1].plot(time_cost4, test_acc4,linewidth=0.8) 
ax[1].plot(time_cost2, test_acc2,linewidth=0.8) 
ax[1].plot(time_cost5, test_acc5,linewidth=0.8)
ax[1].plot(time_cost3, test_acc3,linewidth=0.8) 
ax[1].plot(time_cost6, test_acc6,linewidth=0.8)

 
ax[0].legend(["SGD (lr = 0.05)","SGD+FN(lr = 0.05)","SGD (lr = 0.01)","SGD+FN (lr = 0.01)","SGD (lr = 0.005)","SGD+FN (lr = 0.005)"] , loc = "best")  
ax[1].legend(["SGD (lr = 0.05)","SGD+FN(lr = 0.05)","SGD (lr = 0.01)","SGD+FN (lr = 0.01)","SGD (lr = 0.005)","SGD+FN (lr = 0.005)"] , loc = "best")     

pdf.savefig()
plt.show()

plt.close()
pdf.close()



from matplotlib.backends.backend_pdf import PdfPages

 

pdf = PdfPages('./MNIST_MLP_v2.pdf')

fig, ax = plt.subplots(2,3) 
fig.set_figheight(6.5)
fig.set_figwidth(12)
ax[0][0].grid(color='grey',
        linestyle='--',
        linewidth=1,
        alpha=0.3)

ax[0][0].set_ylabel('log-Loss')

ax[0][0].plot(time_cost1, np.log(train_loss1),linewidth=0.8) 
ax[0][0].plot(time_cost4, np.log(train_loss4),linewidth=0.8)


ax[0][1].grid(color='grey',
        linestyle='--',
        linewidth=1,
        alpha=0.3)

ax[0][1].plot(time_cost2, np.log(train_loss2),linewidth=0.8) 
ax[0][1].plot(time_cost5, np.log(train_loss5),linewidth=0.8) 

ax[0][2].grid(color='grey',
        linestyle='--',
        linewidth=1,
        alpha=0.3)

ax[0][2].plot(time_cost3, np.log(train_loss3),linewidth=0.8) 
ax[0][2].plot(time_cost6, np.log(train_loss6),linewidth=0.8)


ax[1][0].grid(color='grey',
        linestyle='--',
        linewidth=1,
        alpha=0.3)
ax[1][0].set_xlabel('Time(second)')
ax[1][0].set_ylabel('Accuracy')
ax[1][0].plot(time_cost1, test_acc1,linewidth=0.8) 
ax[1][0].plot(time_cost4, test_acc4,linewidth=0.8) 


ax[1][1].grid(color='grey',
        linestyle='--',
        linewidth=1,
        alpha=0.3)
ax[1][1].set_xlabel('Time(second)')
ax[1][1].plot(time_cost2, test_acc2,linewidth=0.8) 
ax[1][1].plot(time_cost5, test_acc5,linewidth=0.8)


ax[1][2].grid(color='grey',
        linestyle='--',
        linewidth=1,
        alpha=0.3)
ax[1][2].set_xlabel('Time(second)')
ax[1][2].plot(time_cost3, test_acc3,linewidth=0.8) 
ax[1][2].plot(time_cost6, test_acc6,linewidth=0.8)

ax[0][0].legend(["SGD (lr = 0.05)","FN(lr = 0.05)"] , loc = "best")  
ax[0][1].legend(["SGD (lr = 0.01)","FN(lr = 0.01)"] , loc = "best") 
ax[0][2].legend(["SGD (lr = 0.005)","FN(lr = 0.005)"] , loc = "best") 
ax[1][0].legend(["SGD (lr = 0.05)","FN(lr = 0.05)"] , loc = "best")  
ax[1][1].legend(["SGD (lr = 0.01)","FN(lr = 0.01)"] , loc = "best") 
ax[1][2].legend(["SGD (lr = 0.005)","FN(lr = 0.005)"] , loc = "best") 

pdf.savefig()
plt.show()

plt.close()
pdf.close()





