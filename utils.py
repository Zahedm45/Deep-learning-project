from torchvision import datasets, transforms
import numpy as np
from sklearn.model_selection import train_test_split

from data import Data


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

    # Flattens the dataset e.g. from (60000, 28, 28) -> (60000, 784)
    # 60,000 images and 28×28 pixes
    X_full = full_train_data.data.numpy().reshape(-1, 784).astype(np.float32) / 255.0
    y_full = full_train_data.targets.numpy()

    # same here
    X_test = test_data.data.numpy().reshape(-1, 784).astype(np.float32) / 255.0
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

    return Data(X_train, y_train, X_val, y_val, X_test, y_test)


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

    # Flatten images: (N, 32, 32, 3) -> (N, 3072)
    X_full = X_full.reshape(-1, 32 * 32 * 3)
    X_test = X_test.reshape(-1, 32 * 32 * 3)

    # Split train into train + validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_full, y_full,
        test_size=0.17,
        random_state=42,
        shuffle=True
    )

    # One-hot encode targets
    y_train = encode_one_hot(y_train, 10)
    y_val   = encode_one_hot(y_val, 10)
    y_test  = encode_one_hot(y_test, 10)

    return Data(X_train, y_train, X_val, y_val, X_test, y_test)
