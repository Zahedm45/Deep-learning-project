
class Data():
    def __init__(self, X_train, y_train, X_val, y_val, X_test, y_test, n_features, n_classes):
        self.X_train = X_train
        self.y_train = y_train

        self.X_val = X_val
        self.y_val = y_val

        self.X_test = X_test
        self.y_test = y_test

        self.n_features = n_features
        self.n_classes = n_classes

