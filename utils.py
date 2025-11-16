from torchvision import datasets, transforms
import numpy as np
from sklearn.model_selection import train_test_split





def load_fashion_mnist_torch():
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

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def encode_one_hot(y: np.ndarray, num_classes: int) -> np.ndarray:
    return np.eye(num_classes)[y]

def decode_one_hot(y_one_hot: np.ndarray) -> np.ndarray:
    return np.argmax(y_one_hot, axis=1)