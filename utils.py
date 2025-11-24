from torchvision import datasets, transforms # mnist and cifar
from sklearn.datasets import load_iris # iris
import numpy as np
from sklearn.model_selection import train_test_split
from data import Data
from sklearn.preprocessing import StandardScaler



def load_fashion_mnist():
    transform = transforms.Compose([
        transforms.ToTensor()
    ])


    full_train_data = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=transform
    )

    test_data = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=transform
    )

    n_classes = len(full_train_data.classes)
    n_features = 28 * 28

    # Flattens the dataset e.g. from (60000, 28, 28) -> (60000, 784)
    # 60,000 images and 28×28 pixes

    X_full = full_train_data.data.numpy().reshape(-1, n_features).astype(np.float32) / 255.0
    y_full = full_train_data.targets.numpy()

    # same here
    X_test = test_data.data.numpy().reshape(-1, n_features).astype(np.float32) / 255.0
    y_test = test_data.targets.numpy()

    # Split train into train + validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_full, y_full,
        test_size=0.17,  # ≈15% of total dataset
        random_state=42,
        shuffle=True
    )

    # Encodes the target values
    y_train = encode_one_hot(y_train, 10)
    y_val   = encode_one_hot(y_val, 10)
    y_test  = encode_one_hot(y_test, 10)

    return Data(X_train, y_train, X_val, y_val, X_test, y_test, n_features, n_classes)


def encode_one_hot(y: np.ndarray, num_classes: int) -> np.ndarray:
    return np.eye(num_classes)[y]

def decode_one_hot(y_one_hot: np.ndarray) -> np.ndarray:
    return np.argmax(y_one_hot, axis=1)



def load_cifar10():
    transform = transforms.Compose([
        transforms.ToTensor()      # Converts to (C,H,W) in [0,1]
    ])

    full_train_data = datasets.CIFAR10(
        root="data",
        train=True,
        download=True,
        transform=transform
    )

    test_data = datasets.CIFAR10(
        root="data",
        train=False,
        download=True,
        transform=transform
    )


    X_full = full_train_data.data.astype(np.float32) / 255.0   # shape (50000, 32, 32, 3)
    y_full = np.array(full_train_data.targets)

    X_test = test_data.data.astype(np.float32) / 255.0
    y_test = np.array(test_data.targets)

    n_features = 32 * 32 * 3
    n_classes = len(full_train_data.classes)

    # Flatten images: (N, 32, 32, 3) -> (N, 3072)
    X_full = X_full.reshape(-1, n_features)
    X_test = X_test.reshape(-1, n_features)

    # Split train into train + validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_full, y_full,
        test_size=0.17,
        random_state=42,
        shuffle=True
    )

    # One-hot encode targets
    y_train = encode_one_hot(y_train, n_classes)
    y_val   = encode_one_hot(y_val, n_classes)
    y_test  = encode_one_hot(y_test, n_classes)

    return Data(
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        X_test=X_test, y_test=y_test,
        n_features=n_features,
        n_classes=n_classes
    )




def load_iris_dataset():
    iris = load_iris()

    X = iris.data.astype(np.float32)     # shape (150, 4)
    y = iris.target                       # shape (150,)

    scaler = StandardScaler()
    X = scaler.fit_transform(X).astype(np.float32)


    n_features = X.shape[1]
    n_classes = len(np.unique(y))

    # Train/val/test split (70/15/15)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, shuffle=True
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, shuffle=True
    )

    # One-hot encode targets (reuse your function)
    y_train = encode_one_hot(y_train, n_classes)
    y_val   = encode_one_hot(y_val, n_classes)
    y_test  = encode_one_hot(y_test, n_classes)

    return Data(
        X_train=X_train, y_train=y_train,
        X_val=X_val,     y_val=y_val,
        X_test=X_test,   y_test=y_test,
        n_features=n_features,
        n_classes=n_classes
    )

















