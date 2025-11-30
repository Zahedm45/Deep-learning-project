
import numpy as np

class Acti:
    relu = "relu"
    tanh = "tanh"
    sigmoid = "sigmoid"
    as_arr = [relu, tanh, sigmoid]


def rectified_linear_unit(x):
    return np.maximum(0, x)

def rectified_linear_unit_derivative(x):
    return (x > 0).astype(float)


def hyperbolic_tan(x):
    return np.tanh(x)

def hyperbolic_tan_derivative(x):
    return 1 - np.tanh(x) ** 2


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))



def softmax(z):
    z = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)



# Activation functions map
activations_map = {
    Acti.relu: (rectified_linear_unit, rectified_linear_unit_derivative),
    Acti.tanh: (hyperbolic_tan, hyperbolic_tan_derivative),
    Acti.sigmoid: (sigmoid, sigmoid_derivative),
}