
import numpy as np

# Stochastic Gradient Descent optimizer.
class SGD:
    def __init__(self, learning_rate: float = 0.01):
        self.learning_rate = learning_rate

    def update(self, params, grads):
        for key in params:
            params[key] -= self.learning_rate * grads[key]



