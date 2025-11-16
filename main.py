from nn.FFNN import FFNN
from utils import load_fashion_mnist_torch

if __name__ == "__main__":
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_fashion_mnist_torch()

    ffnn_model = FFNN(num_features=784, num_output=10, num_hidden_layers=2, hidden_units=128,
                      activation="relu", weight_init="he", optimizer="adam", learning_rate=0.001,
                      l2_coeff=1e-4, batch_size=128, epochs=10
                      )
