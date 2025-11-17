import run
from data import Data
from nn.FFNN import FFNN
from utils import load_fashion_mnist, load_cifar10

if __name__ == "__main__":
    # mnist_data: Data = load_fashion_mnist()
    #
    # ffnn_model = FFNN(n_features=784, n_output_ne=10, n_hid_layers=2, n_hid_neurons=128,
    #                   activation="relu", weight_init="he", optimizer="adam", learning_rate=0.001,
    #                   l2_coeff=1e-4, batch_size=128, epochs=10
    #                   )
    #
    # run.train(ffnn_model, mnist_data.X_train, mnist_data.y_train, mnist_data.X_val, mnist_data.y_val)
    #
    # run.evaluate(ffnn_model, mnist_data.X_test, mnist_data.y_test)



    cifa_data: Data = load_cifar10()
    ffnn_model2 = FFNN(n_features=3072, n_output_ne=10, n_hid_layers=2, n_hid_neurons=128,
                      activation="relu", weight_init="he", optimizer="adam", learning_rate=0.001,
                      l2_coeff=1e-4, batch_size=128, epochs=20
                      )

    run.train(ffnn_model2, cifa_data.X_train, cifa_data.y_train, cifa_data.X_val, cifa_data.y_val, 5)

    run.evaluate(ffnn_model2, cifa_data.X_test, cifa_data.y_test)