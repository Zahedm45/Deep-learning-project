from nn.activation_functions import activations_map, softmax
import numpy as np
from jaxtyping import Float
from nn.optimizers import Adam, SGD


class FFNN:


    def __init__(
        self,
        num_features: int,
        num_output: int,                    # number of output neurons
        num_hidden_layers: int = 2,
        hidden_units: int = 128,
        activation: str = "relu",
        weight_init: str = "he",
        optimizer: str = "adam",
        learning_rate: float = 0.001,
        l2_coeff: float = 0.0,
        batch_size: int = 128,
        epochs: int = 20,
        seed: int = 42,
    ):
        self.cache = {}
        np.random.seed(seed)

        self.num_hid_layers = num_hidden_layers
        self.hidden_units = hidden_units
        self.activation_name = activation
        self.weight_init = weight_init
        self.batch_size = batch_size
        self.epochs = epochs
        self.l2_coeff = l2_coeff


        self.activation, self.activation_derivative = activations_map[activation]

        # Optimizer
        if optimizer == "adam":
            self.optimizer = Adam(learning_rate)
        else:
            self.optimizer = SGD(learning_rate)


        # Creating an array containing number of neurons in each layer.
        input_layer = [num_features]
        hidden_layers = [hidden_units] * num_hidden_layers
        output_layer = [num_output]
        # Concatenation, so one array like [10, 128, 128, 3, ..., total layers]
        layer_dims = input_layer + hidden_layers + output_layer
        
        # Init weights W1, W2, ..., WD and bias b
        self.params = {} # parameter map
        for i in range(len(layer_dims) - 1):
            self.params[f"W{i+1}"] = init_weights(layer_dims[i], layer_dims[i + 1], method=weight_init)
            self.params[f"b{i+1}"] = np.zeros((1, layer_dims[i+1])) # bias b




    def feed_forward(self, X: np.ndarray) -> np.ndarray:
        self.cache = {"A0": X}

        # Hidden layers
        for i in range(1, self.num_hid_layers + 1):
            A_prev = self.cache[f"A{i - 1}"]  # previous layer output
            W = self.params[f"W{i}"]  # weight matrix for this layer
            b = self.params[f"b{i}"]  # bias vector for this layer
            weighted_sum = np.matmul(A_prev, W)  # matrix multiplication
            Z = weighted_sum + b

            self.cache[f"Z{i}"] = Z
            self.cache[f"A{i}"] = self.activation(Z)

        # Output layer
        A_prev = self.cache[f"A{self.num_hid_layers}"]  # previous layer activation
        W_out = self.params[f"W{self.num_hid_layers + 1}"]  # output layer weight matrix
        b_out = self.params[f"b{self.num_hid_layers + 1}"]  # Output layer bias vector

        weighted_sum_out = np.matmul(A_prev, W_out)  # matrix multiplication
        Z_out = weighted_sum_out + b_out

        self.cache[f"Z{self.num_hid_layers + 1}"] = Z_out
        A_out = softmax(Z_out)
        self.cache[f"A{self.num_hid_layers + 1}"] = A_out
        return A_out



    # Does backpropagation and updates weights.
    def backpropagate(self, y_true: np.ndarray):
        gradients = {}
        m = y_true.shape[0]
        L = self.num_hid_layers + 1

        dZ = self.cache[f"A{L}"] - y_true
        for i in reversed(range(1, L + 1)):
            A_prev = self.cache[f"A{i-1}"]

            # computes derivative dW = ∂L/∂W for layer i (including L2 regularization)
            A_prev_T = A_prev.T                         # transpose
            dW_no_reg = np.matmul(A_prev_T, dZ) / m     # gradient without L2
            W_reg_term = self.l2_coeff * self.params[f"W{i}"]  # L2 regularization term
            dW = dW_no_reg + W_reg_term
            gradients[f"W{i}"] = dW

            # Computes derivative db = ∂L/∂b by summing dZ across the batch
            db = np.sum(dZ, axis=0, keepdims=True) / m
            gradients[f"b{i}"] = db

            # Backpropagate dZ to the previous layer (skip when i == 1, since A0 is input)
            if i > 1:
                dA_prev = np.matmul(dZ, self.params[f"W{i}"].T)
                dZ = dA_prev * self.activation_derivative(self.cache[f"Z{i-1}"])

        self.optimizer.update(self.params, gradients)



#Initializes weights using He initialization.
def init_weights(in_features, out_features, method="he") -> Float[np.ndarray, "in_feat out_feat"]:
    if method == "he":
        scale = np.sqrt(2.0 / in_features)
    # TODO: Add another wieght init method
    else:
        scale = 0.01
    return np.random.randn(in_features, out_features) * scale












