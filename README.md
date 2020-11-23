## Introduction

This repository contains codes for *Factor Normalization for Deep Neural Network Models* 

## Details of experiments

### About experiment environment
All codes are  based on Python language and are based on tensorflow 2.2.0. We run all codes on a Tesla K80 GPU with 11GB memory.

### About model initialization

We first initialize an AlexNet and a ResNet50(from keras.applications.resnet50), then we save the initialized model weights to an h5 file to ensure all experiments are based on the same initialization.

### About factor decomposition

**1. high dimensional SVD**

We calculate the factor weight by conducting SVD on a large number of training data in advanced. Standard PCA can not directly conducted because parameter dimension p is extremely large. However, SVD allows us to calculate row sigular vector first (we can choose sample size N relative small compared to p), then eliminate left orthogonal matrix and obtain column sigular vector. Then we save the factor weights and regression coefficients (for separating residual) to an npz file so that we don't need to estimate the factor weights every time. 

**2. residual regression**

Once we receive new data, we load weights from npz file and calculate factor. Then we conduct residual regression to remove the factor effect from original feature.
