from nn.FFNN import FFNN
from nn.Initializers import Init
from nn.activation_functions import Acti
from nn.optimizers import Optim

sweep_config1 = {
    "method": "bayes",
    "metric": {"name": "val_acc", "goal": "maximize"},
    "parameters": {
        "n_hid_layers": {"values": [1, 2, 3]},
        "n_hid_neurons": {"values": [64, 128, 256, 512, 1024]},
        "learning_rate": {"min": 0.0001, "max": 0.05},
        "l2_coeff": {"values": [0.0, 0.00001, 0.0001, 0.0005, 0.001]},
        "batch_size": {"values": [64, 128, 256, 512]},
        "activation": {"values": Acti.as_arr},
        "optimizer": {"values": Optim.as_arr},
        "weight_init": {"values": Init.as_arr},
        "epochs": {"value": 15},
        "dropout_rate": {
            "min": 0.0,
            "max": 0.4
        }
    }
}


sweep_config2 = {
    "method": "bayes",
    "metric": {"name": "val_acc", "goal": "maximize"},
    "parameters": {
        "n_hid_layers": {"values": [1, 2, 3]},
        "n_hid_neurons": {"values": [64, 128, 256]},
        "learning_rate": {"min": 1e-4, "max": 1e-2},
        "l2_coeff": {"values": [0.0, 1e-4, 1e-3]},
        "batch_size": {"values": [64, 128, 256]},
        "activation": {"values": [Acti.sigmoid]},
        "optimizer": {"values": Optim.as_arr},
        "weight_init": {"values": Init.as_arr},
        "epochs": {"value": 15}
    }
}



cifar1 = {
    "method": "bayes",
    "metric": {"name": "val_acc", "goal": "maximize"},

    "parameters": {
        "n_hid_layers": {"values": [2, 3, 4]},
        "n_hid_neurons": {"values": [256, 512, 768, 1024]},
        "activation": {"values": Acti.as_arr},
        "optimizer": {"values": Optim.as_arr},
        "weight_init": {"values": Init.as_arr},
        "learning_rate": {
            "distribution": "log_uniform_values",
            "min": 1e-4,
            "max": 5e-3
        },
        "l2_coeff": {"values": [0.0, 1e-5, 1e-4, 5e-4, 1e-3]},
        "dropout": {"values": [0.0, 0.1, 0.2, 0.3, 0.5]},
        "batch_size": {"values": [64, 128, 256]},
        "epochs": {"value": 40}
    }
}
def best_fnm(mn_data):
    return FFNN(
        n_features=mn_data.n_features,
        n_output_ne=mn_data.n_classes,
        n_hid_layers=3,
        n_hid_neurons=512,
        activation=Acti.tanh,
        weight_init=Init.xavier,
        optimizer=Optim.adam,
        learning_rate=0.0006443732449600567,
        l2_coeff=0,
        batch_size=512,
        epochs=15,
        dropout_rate=0.051626532673010095
    )


def best_cifar(ci_data):
    return FFNN(
        n_features=ci_data.n_features,
        n_output_ne=ci_data.n_classes,
        n_hid_layers=2,
        n_hid_neurons=256,
        activation="relu",
        weight_init="he",
        optimizer="adam",
        learning_rate=0.001,
        l2_coeff=1e-4,
        batch_size=128,
        epochs=20,
        dropout_rate=0.0
    )