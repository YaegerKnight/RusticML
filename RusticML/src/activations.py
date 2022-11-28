"""
description: This file contains all the functions required to activate a layer
"""
import numpy as np
from core import Tensor

def Softmax(x: Tensor):

    exp = np.exp(x.value - np.max(x.value))
    result = Tensor(exp / exp.sum()) 

    return result

def tanh(x: Tensor):

    output = (np.exp(x.value) - np.exp(-x.value))/(np.exp(x.value) + np.exp(-x.value))
    result = Tensor(value=output, parents=(x,))
    
    def _backward():
        x.gradient +=  (1 - (output ** 2)) * result.gradient
    
    result._backward = _backward
    return result

def ReLU(x: Tensor):

    output = np.max(x.value, 0)
    result = Tensor(value=output, parents=(x,))

    def _backward():
        x.gradient += (int((output*1) >0)) * result.gradient
    
    result._backward = _backward

    return result

def Sigmoid(x: Tensor):

    output = 1/(1 + np.exp(-x.value))
    result = Tensor(value=output, parents=(x,))

    def _backward():
        x.gradient += 1.0 * output * (1 - output) * result.gradient
    result._backward = _backward

    return result