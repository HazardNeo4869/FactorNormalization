# 载入必要的库
import os
import time
import numpy as np
import pandas as pd
import random as random


import tensorflow as tf
from keras.models import load_model
from keras import Model
from keras.preprocessing.image import ImageDataGenerator

from FN import Estimate_FactorX
from FN import Separate_FactorX
from keras.layers import (Activation, Conv2D, Dense, Dropout, Flatten,BatchNormalization,
                          GlobalAveragePooling2D, Input, MaxPooling2D, add,concatenate)
from keras import regularizers
from keras import Model


def fn_model(old_model,num_class):
    
    # 第一步 去掉old_model的最后一层
    
    num_layers =  len(old_model.layers)    # 读取输入模型一共有多少层
    print("Old model has " + str(num_layers) + " layers in total! ")
    model0 = Model(inputs=old_model.input, outputs=old_model.layers[num_layers-2].output)
    
    # 第二步 构造多输入的新模型
    input_z = Input([1])
    output_z = input_z
    model1 = Model(input_z, output_z)
    
    combined = concatenate([model0.output, model1.output])
    #combined = BatchNormalization()(combined)
    
    z = Dense(num_class, activation="softmax")(combined)
    
    new_model = Model(inputs=[model0.input, model1.input], outputs=z)
    
    return new_model


'''
直接整理数据：
'''
# 读取
cifar10 = tf.keras.datasets.cifar10
(x0, y0), (X1, Y1) = cifar10.load_data()
# 整理格式
N0=x0.shape[0];N1=X1.shape[0]
x0 = x0.reshape(N0,32,32,3)/255.0
X1 = X1.reshape(N1,32,32,3)/255.0
x0 = x0.astype(np.float32)
X1 = X1.astype(np.float32)


datagen = ImageDataGenerator(
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    shear_range=0.5,
    zoom_range=0.2, 
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True).flow(x0,y0,batch_size = 50000)


Epochs = 200
batch_size = 200
K0 = int(N0/batch_size)
K1 = int(N1/batch_size)


model = load_model("./alexnet_init.h5")

time_cost = np.zeros(Epochs)
time_checker = 0
loss = np.zeros(Epochs)
val_loss = np.zeros(Epochs)
acc = np.zeros(Epochs)
val_acc = np.zeros(Epochs)
    
metrics = tf.keras.metrics.SparseCategoricalAccuracy()
for i in range(Epochs):
    
    
    optimizer = tf.keras.optimizers.SGD(learning_rate = 0.1/np.sqrt(i+1))
    train_loss_avg = 0.
    test_loss_avg = 0.
    train_acc_avg = 0.
    test_acc_avg = 0.
    
    # 重新生成全样本
    X0,Y0 = next(datagen)
    
    start = time.time()
    # 训练
    for k in range(K0):

        X,Y = X0[k*batch_size:(k+1)*batch_size,],Y0[k*batch_size:(k+1)*batch_size].reshape(batch_size,)
        with tf.GradientTape() as tape:
            Y_pred = model(X)
            Loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=Y, y_pred=Y_pred)
            Loss = tf.reduce_mean(Loss)
        grads = tape.gradient(Loss, model.trainable_variables)
        optimizer.apply_gradients(grads_and_vars=zip(grads, model.trainable_variables))
        
        train_loss_avg = (k/(k+1)) * train_loss_avg + (1/(k+1)) * Loss.numpy()
        metrics.reset_states()
        metrics.update_state(y_true=Y, y_pred=Y_pred)
        train_acc_avg = (k/(k+1)) * train_acc_avg + (1/(k+1)) * metrics.result().numpy() 
        print("\rEpoch: {:d} batch: {:d} loss: {:.4f} acc: {:.4f} | {:.2%}"
                .format(i+1, k+1, train_loss_avg, train_acc_avg, (k+1)*1.0/K0), end='',  flush=True)
        
    end = time.time()
    time_checker = time_checker + end-start
    time_cost[i] = time_checker
 
    loss[i] = train_loss_avg
    acc[i] = train_acc_avg
    
    # 进行验证
    for k in range(K1):
        X,Y = X1[k*batch_size:(k+1)*batch_size,],Y1[k*batch_size:(k+1)*batch_size].reshape(batch_size,)
        Y_pred = model(X)                                                          # 通过调用model，而不是显式表达
        Loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=Y, y_pred=Y_pred)   # 损失采用cross_entropy，是一个ss*1的向量
        Loss = tf.reduce_mean(Loss)                                                # 每一维求均值得到均值版本的loss
        test_loss_avg = test_loss_avg + Loss.numpy()
        metrics.reset_states()
        metrics.update_state(y_true=Y, y_pred=Y_pred)
        test_acc_avg = test_acc_avg + metrics.result().numpy()    
    
    val_loss[i] = test_loss_avg/K1
    val_acc[i] = test_acc_avg/K1
    
    print("\rEpoch: {:d}/{:d} | loss: {:.4f} acc: {:.4f} | val_loss: {:.4f}  val_acc: {:.4f} | time: {:.2f}s"
            .format(i+1, Epochs, loss[i], acc[i], val_loss[i], val_acc[i], end-start), end='\n')

df = pd.DataFrame([time_cost,loss,acc,val_loss,val_acc])
new_col = ["time_cost","loss","acc","val_loss","val_acc"]
df2 = pd.DataFrame(df.values.T, columns=new_col)
df2.to_csv('./AlexNet_SGD(1117).csv', index=False, header=True)


# 读取权重
data = np.load("weights.npz")
fweight = data["fweight"]
beta = data["beta"]
s = data["s"]

ratio = (s[0]/(s[1]+1e-7))**2
#Xw = tf.concat([tf.ones([2048,1]),(1/ratio) * tf.ones([1,1])],0)
Xw = tf.concat([tf.ones([4096,1]),(1/ratio) * tf.ones([1,1])],0)
X1 = tf.reshape(X1,[N1,3072])
X11,Z1 = Separate_FactorX(X1,fweight,beta)
X11 = tf.reshape(X11,[N1,32,32,3])

model = load_model("./alexnet_init.h5")

model_fn = fn_model(model,10)


time_cost = np.zeros(Epochs)
time_checker = 0

loss = np.zeros(Epochs)
val_loss = np.zeros(Epochs)

acc = np.zeros(Epochs)
val_acc = np.zeros(Epochs)
    
    
metrics = tf.keras.metrics.SparseCategoricalAccuracy()

for i in range(Epochs):
    
    
    optimizer = tf.keras.optimizers.SGD(learning_rate = 0.15/np.sqrt(i+1))
    train_loss_avg = 0.
    test_loss_avg = 0.
    train_acc_avg = 0.
    test_acc_avg = 0.
    
    # 重新生成全样本
    X_old,Y0 = next(datagen)
    X_old = tf.reshape(X_old,[50000,3072])
    start = time.time()
    X0,Z0 = Separate_FactorX(X_old,fweight,beta)
    X0 = tf.reshape(X0,[50000,32,32,3])
    
    # 训练
    for k in range(K0):

        X,Y = X0[k*batch_size:(k+1)*batch_size,],Y0[k*batch_size:(k+1)*batch_size].reshape(batch_size,)
        Z = Z0[k*batch_size:(k+1)*batch_size,]
        with tf.GradientTape() as tape:
            Y_pred = model_fn([X,Z])
            Loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=Y, y_pred=Y_pred)
            Loss = tf.reduce_mean(Loss)
        grads = tape.gradient(Loss, model_fn.trainable_variables)
        grads[len(grads)-2] = Xw * grads[len(grads)-2]
            #grads[len(grads)-2] = Xw * grads[len(grads)-2]; grads[len(grads)-1] = bw * grads[len(grads)-1]
        optimizer.apply_gradients(grads_and_vars=zip(grads, model_fn.trainable_variables))
        
        train_loss_avg = (k/(k+1)) * train_loss_avg + (1/(k+1)) * Loss.numpy()
        metrics.reset_states()
        metrics.update_state(y_true=Y, y_pred=Y_pred)
        train_acc_avg = (k/(k+1)) * train_acc_avg + (1/(k+1)) * metrics.result().numpy() 
        print("\rEpoch: {:d} batch: {:d} loss: {:.4f} acc: {:.4f} | {:.2%}"
                .format(i+1, k+1, train_loss_avg, train_acc_avg, (k+1)*1.0/K0), end='',  flush=True)
        
    end = time.time()
    time_checker = time_checker + end-start
    time_cost[i] = time_checker
 
    loss[i] = train_loss_avg
    acc[i] = train_acc_avg
    
    # 进行验证
    for k in range(K1):
        X,Y = X11[k*batch_size:(k+1)*batch_size,],Y1[k*batch_size:(k+1)*batch_size].reshape(batch_size,)
        Z = Z1[k*batch_size:(k+1)*batch_size,]
        Y_pred = model_fn([X,Z])                                                          # 通过调用model，而不是显式表达
        Loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=Y, y_pred=Y_pred)   # 损失采用cross_entropy，是一个ss*1的向量
        Loss = tf.reduce_mean(Loss)                                                # 每一维求均值得到均值版本的loss
        test_loss_avg = test_loss_avg + Loss.numpy()
        metrics.reset_states()
        metrics.update_state(y_true=Y, y_pred=Y_pred)
        test_acc_avg = test_acc_avg + metrics.result().numpy()    
    
    val_loss[i] = test_loss_avg/K1
    val_acc[i] = test_acc_avg/K1
    
    print("\rEpoch: {:d}/{:d} | loss: {:.4f} acc: {:.4f} | val_loss: {:.4f}  val_acc: {:.4f} | time: {:.2f}s"
            .format(i+1, Epochs, loss[i], acc[i], val_loss[i], val_acc[i], end-start), end='\n')


df = pd.DataFrame([time_cost,loss,acc,val_loss,val_acc])
new_col = ["time_cost","loss","acc","val_loss","val_acc"]
df2 = pd.DataFrame(df.values.T, columns=new_col)
df2.to_csv('./AlexNet_FN_SGD(1117).csv', index=False, header=True)