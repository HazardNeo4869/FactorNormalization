# FactorNormalization
code for *Factor Normalization for Deep  Neural Network Models*

## About model initialization
We first initialize an AlexNet and a ResNet50, then we save the initialized model weights to ensure all experiments are based on same initialization.

## About factor weight and other coefficients
We calculate the factor weight by conducting SVD on a large number of training data in advanced. Then we save the factor weights and regression coefficients (for separating residual) to an npz file so that we don't need to estimate the factor weights every time. 
