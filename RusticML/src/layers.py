"""
description: This class holds the basic layer components required to build a neural network
"""
import abc
import numpy as np
import dill as pk
from core import *
from activations import *


class BaseModule(metaclass = abc.ABCMeta):

    
    def __init__(self) -> None:
        pass

    def __call__(self):
        pass

    def parameters(self):
        pass
    
    def gradients_to_zero(self):
        pass

    def save(self,filename):
        try:
            if filename is None or filename == "":
                raise ValueError("Empty filename. Cannot save model")
        except ValueError as err:
            print(err)
            return

        pk.dump(self, open(f'{filename}', 'wb'))
        print(f"Model saved as {filename} in current directory")
    
    def load(filename):
        try:
            if filename is None or filename == "":
                raise ValueError("Empty filename. Cannot load model")
        except ValueError as err:
            print(err)
            return
        model = pk.load(open(f'{filename}', 'rb'))
        print(f"Loaded model {filename} in current directory")
        return model

class Linear(BaseModule):

    def __init__(self, num, activation=None) -> None:
        super().__init__()
        self.weights = [Tensor(value=np.random.rand()) for i in range(num)]
        self.bias = Tensor(value = np.random.rand())
        self.name = "Linear"
        self.activation = activation
    
    def __call__(self, x):
        
        """
        Define the forward pass for a linear layer i.e y = wx + b
        """
        _sum = 0
        #  sum((wi*xi for wi,xi in zip(self.weights, x)), self.bias)
        for w,x in zip(self.weights, x):
            res = w*x
            _sum += res
        _sum += self.bias
        return tanh(_sum)
    
    def parameters(self):
        return self.weights + [self.bias] 

    def gradients_to_zero(self):
        for p in self.parameters():
            p.gradient = 0

    def __repr__(self) -> str:
        return f"Layer({self.name}, params={len(self.parameters())})"

class Dense(BaseModule):

    def __init__(self, num_input, num_output) -> None:
        super().__init__()
        self.nodes = [Linear(num_input) for _ in range(num_output)]
        self.name = "Dense"
    
    def __call__(self, x):
        output =  [n(x) for n in self.nodes]
        return output   

    def parameters(self):
        
        params_list = []
        for node in self.nodes:
            parameters = node.parameters()
            for item in parameters:
                params_list.append(item)
        return params_list
    
    def gradients_to_zero(self):
        for p in self.parameters():
            p.gradient = 0

class MLP(BaseModule):

    def __init__(self, num_input, num_output) -> None:
        super().__init__()
        sz = [num_input] + num_output
        self.layers = [Dense(sz[i], sz[i+1]) for i in range(len(num_output))]
        self.name = "Multi-Layer Perceptron"
    
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        
        params_list = []
        for layer in self.layers:
            parameters = layer.parameters()
            for item in parameters:
                params_list.append(item)
        return params_list
    
    def gradients_to_zero(self):
        params = self.parameters()
        for p in params:
            p.gradient = 0
    
    