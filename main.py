from operator import irshift

import run
from config import sweep_config1, sweep_config2, cifar1, best_fnm, best_cifar
from data import Data
from nn.FFNN import FFNN
from nn.Initializers import Init
from nn.activation_functions import Acti
from nn.optimizers import Optim
from utils import load_fashion_mnist, load_cifar10, load_iris_dataset

import wandb


def sweep_train(data):
    wandb.init()
    cfg = wandb.config

    model = FFNN(
        n_features=data.n_features,
        n_output_ne=data.n_classes,
        n_hid_layers=cfg.n_hid_layers,
        n_hid_neurons=cfg.n_hid_neurons,
        activation=cfg.activation,
        weight_init=cfg.weight_init,
        optimizer=cfg.optimizer,
        learning_rate=cfg.learning_rate,
        l2_coeff=cfg.l2_coeff,
        batch_size=cfg.batch_size,
        epochs=cfg.epochs
    )

    run.train(
        model,
        data.X_train, data.y_train,
        data.X_val, data.y_val,
    )




def cifa():
    ci_data: Data = load_cifar10()
    # sweep_id = wandb.sweep(cifar1, project="cifar")
    # wandb.agent(sweep_id, function=lambda: sweep_train(ci_data), count=30)

    ci_model = best_cifar(ci_data)

    run.train(ci_model, ci_data.X_train, ci_data.y_train, ci_data.X_val, ci_data.y_val, 15)
    run.evaluate(ci_model, ci_data.X_test, ci_data.y_test)

def mnist():
    mn_data: Data = load_fashion_mnist()

    # sweep_id = wandb.sweep(sweep_config1, project="fmn")
    # wandb.agent(sweep_id, function=lambda: sweep_train(mn_data), count=40)

    mn_model = best_fnm(mn_data)

    run.train(mn_model, mn_data.X_train, mn_data.y_train, mn_data.X_val, mn_data.y_val, patience=15)
    run.evaluate(mn_model, mn_data.X_test, mn_data.y_test)



def irish():

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



if __name__ == "__main__":
    #wandb.login(key="80e34afedacdbb1d88db7ef60f755b6b7666eb4e")
    mnist()
    # cifa()

