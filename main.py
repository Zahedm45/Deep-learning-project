

from torchvision import datasets
import numpy as np

# Download Fashion-MNIST
train_data = datasets.FashionMNIST(root='data/fashion_mnist', train=True, download=True)
test_data  = datasets.FashionMNIST(root='data/fashion_mnist', train=False, download=True)


# Convert to NumPy and normalize
X_train = train_data.data.numpy().astype(np.float32) / 255.0
y_train = train_data.targets.numpy()
X_test  = test_data.data.numpy().astype(np.float32) / 255.0
y_test  = test_data.targets.numpy()


#print(X_train.shape)

# Flatten
X_train = X_train.reshape(len(X_train), -1)
X_test  = X_test.reshape(len(X_test), -1)

#print(X_train[:1])

print(X_train.shape)

# print("Train:", X_train.shape, y_train.shape)
# print("Test:", X_test.shape, y_test.shape)


print("Done")

