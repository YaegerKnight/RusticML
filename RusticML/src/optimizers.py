"""
This class contains the optimizers that can be used to optimize the weights of the neural network.
"""
import abc

class BaseOptimizer(metaclass=abc.ABCMeta):

    def __init__(self, params, lr = 0.001) -> None:
        self.params = params
        self.lr = lr

    def optimize(self):
        pass


class StochasticGradientDescent(BaseOptimizer):

    def __init__(self, params, lr) -> None:
        super().__init__(params, lr)
    
    def optimize(self):
        for p in self.params:
            p.value += -self.lr * p.gradient

