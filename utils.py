import numpy as np
import matplotlib.pyplot as plt
import h5py

def load_data():
    train_dataset= h5py.File('train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) #train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) #train set labels
    
    test_dataset= h5py.File('test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) #train set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) #train set labels
    
    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

#initializing 2 layer neural network
def initialize_parameters(n_x,n_h,n_y):
    np.random.seed(1)
    W1=np.random.randn(n_h,n_x)*0.01
    b1=np.zeros((n_h,1))
    W2=np.random.randn(n_y,n_h)*0.01
    b2=np.zeros((n_y,1))
  
    assert(W1.shape==(n_h,n_x))
    assert(b1.shape==(n_h,1))
    assert(W2.shape==(n_y,n_h))
    assert(b2.shape==(n_y,1))
  
    parameters={"W1":W1,
              "b1":b1,
              "W2":W2,
              "b2":b2}
    return parameters


#initializing  L layer neural network
def initialize_parameters_deep(layer_dims):
    np.random.seed(3)
    parameters = {}
    L = len(layer_dims)
    for l in range(1,L):
        parameters["W" + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1])*0.01
        parameters["b" + str(l)] = np.zeros((layer_dims[l], 1))
    
        assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
        assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))
    return parameters

#Forward Propagation module

#linear_forward
def linear_forward(A, W, b):
    Z=np.dot(W,A)+b
    assert(Z.shape==(W.shape[0],A.shape[1]))
    cache=(A,W,b)
    return Z, cache

def sigmoid(Z):
    A= 1/(1+np.exp(-Z))
    cache = Z
    return A, cache

def relu(Z):
    A=np.maximum(0,Z)
    assert(A.shape == Z.shape)
    cache = Z
    return A,cache

#linear_activation_forward
def linear_activation_forward(A_prev, W, b, activation):
    if activation == "sigmoid":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)
    
    elif activation=="relu":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)
    
    assert(A.shape==(W.shape[0],A_prev.shape[1]))
    cache = (linear_cache, activation_cache)
  
    return A, cache

#L model forward
def L_model_forward(X, parameters):
    caches =[]
    A = X
    L = len(parameters)//2\
  
  
    for l in range(1,L):
        A_prev = A
        A, cache = linear_activation_forward(A_prev, parameters["W"+ str(l)], parameters["b"+ str(l)], "relu")
        caches.append(cache)
    
    
    AL, cache = linear_activation_forward(A, parameters["W" + str(L)], parameters["b" + str(L)], "sigmoid")
    caches.append(cache)
  
    assert(AL.shape == (1,X.shape[1]))
    return AL, caches

#Cost Function

def compute_cost(AL, Y):
    m= Y.shape[1]
    cost = -(1/m)*np.sum(Y*np.log(AL) + (1-Y)* np.log(1-AL))
    cost = np.squeeze(cost)
    assert(cost.shape == ())
    return cost

#Backward Propagation Module


def linear_backward(dZ, cache):
    A_prev, W,b=cache
    m=A_prev.shape[1]
  
    dW= (1/m)* np.dot(dZ,A_prev.T)
    db= (np.sum(dZ,axis = 1, keepdims = True))/m
    dA_prev= np.dot(W.T,dZ)
  
    assert(dW.shape==W.shape)
    assert(db.shape==b.shape)
    assert(dA_prev.shape==A_prev.shape)
  
    return dA_prev, dW, db


def relu_backward(dA, cache):
    Z=cache
    dZ=np.array(dA, copy=True)
    dZ[Z<=0]= 0       # When z <= 0, you should set dz to 0 as well. 
  
    assert(dZ.shape==Z.shape)
    return dZ


def sigmoid_backward(dA, cache):
    Z=cache
    s = 1/(1+np.exp(-Z))
    dZ = dA * s * (1-s)
  
    assert(dZ.shape==Z.shape)
    return dZ


#linear activation backward
def linear_activation_backward(dA, cache, activation):
    linear_cache, activation_cache = cache
  
    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    
    elif activation == "sigmoid":
        dZ= sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    return dA_prev, dW, db

    
#L-model backward
def L_model_backward(AL, Y, caches):
    grads={}
    L = len(caches)
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)
  
  
    dAL= -(np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
  
  
    current_cache =caches[-1]
    grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)]= linear_activation_backward(dAL,       current_cache, "sigmoid") 
  
  
    for l in reversed(range(L-1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA"+str(l+1)], current_cache, "relu")
        grads["dA" + str(l)]= dA_prev_temp
        grads["dW" + str(l+1)] = dW_temp
        grads["db" + str(l+1)] = db_temp
    return grads


#Update Parameters

def update_parameters(parameters, grads, learning_rate):
    L = len(parameters)//2 
    for l in range(L):
        parameters["W" + str(l+1)] = parameters["W"+ str(l+1)] - learning_rate * grads["dW" + str(l+1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * grads["db" + str(l+1)]
    return parameters
    
  



