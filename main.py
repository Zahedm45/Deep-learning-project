from operator import irshift

import run
from data import Data
from nn.FFNN import FFNN
from utils import load_fashion_mnist, load_cifar10, load_iris_dataset

if __name__ == "__main__":


    mn_data: Data = load_fashion_mnist()
    mn_model = FFNN(
        n_features=mn_data.n_features,
        n_output_ne=mn_data.n_classes,
        n_hid_layers=2,
        n_hid_neurons=128,
        activation="relu",
        weight_init="he",
        optimizer="adam",
        learning_rate=0.001,
        l2_coeff=1e-4,
        batch_size=128,
        epochs=10
    )

    run.train(mn_model, mn_data.X_train, mn_data.y_train, mn_data.X_val, mn_data.y_val)
    run.evaluate(mn_model, mn_data.X_test, mn_data.y_test)



    ci_data: Data = load_cifar10()
    ci_model = FFNN(
        n_features=ci_data.n_features,
        n_output_ne=ci_data.n_classes,
        n_hid_layers=2,
        n_hid_neurons=128,
        activation="relu",
        weight_init="he",
        optimizer="adam",
        learning_rate=0.001,
        l2_coeff=1e-4,
        batch_size=128,
        epochs=10
    )

    run.train(ci_model, ci_data.X_train, ci_data.y_train, ci_data.X_val, ci_data.y_val, 5)
    run.evaluate(ci_model, ci_data.X_test, ci_data.y_test)


    ir_data: Data = load_iris_dataset()
    ir_model = FFNN(
        n_features=ir_data.n_features,
        n_output_ne=ir_data.n_classes,
        n_hid_layers=2,
        n_hid_neurons=8,
        activation="relu",
        weight_init="he",
        optimizer="adam",
        learning_rate=0.001,
        l2_coeff=1e-4,
        batch_size=8,
        epochs=100
    )

    run.train(ir_model, ir_data.X_train, ir_data.y_train, ir_data.X_val, ir_data.y_val, 50)
    run.evaluate(ir_model, ir_data.X_test, ir_data.y_test)