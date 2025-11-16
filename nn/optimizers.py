
import numpy as np

# Stochastic Gradient Descent optimizer.
class SGD:
    def __init__(self, learning_rate: float = 0.01):
        self.learning_rate = learning_rate

    def update(self, params, grads):
        for key in params:
            params[key] -= self.learning_rate * grads[key]



# Adam optimizer
class Adam:
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.timestep = 0
        self.momentum = {}
        self.velocity = {}

    def update(self, params, grads):
        self.timestep += 1
        for key in params:
            if key not in self.momentum:
                self.momentum[key] = np.zeros_like(params[key])
                self.velocity[key] = np.zeros_like(params[key])

            self.momentum[key] = self.beta1 * self.momentum[key] + (1 - self.beta1) * grads[key]
            self.velocity[key] = self.beta2 * self.velocity[key] + (1 - self.beta2) * (grads[key] ** 2)

            m_hat = self.momentum[key] / (1 - self.beta1 ** self.timestep)
            v_hat = self.velocity[key] / (1 - self.beta2 ** self.timestep)

            params[key] -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)