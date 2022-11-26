"""
description: This file contains all the functions required to activate a layer
"""
import numpy as np

class Sigmoid():

    def __call__(self, x):
        return 1/(1 + np.exp(-x))
    
    def grad(self, x):
        return self.__call__(x) * (1 - self.__call__(x))

class ReLU():

    def __call__(self, x):
        return np.max(0, x)
    
    def grad(self, x):
        return int((x*1) > 0)

class Softmax(self, x):
    