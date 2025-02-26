class CIFARDataset:
    def __init__(self):
        import tensorflow as tf
        import numpy as np

        from tensorflow.keras.datasets import cifar10
        (X_train, y_train), (X_test, y_test) = cifar10.load_data()

        X_train = X_train.astype("float32") / 255.0
        y_train = np.squeeze(y_train)
        X_test = X_test.astype("float32") / 255.0
        y_test = np.squeeze(y_test)

        self.image_train = X_train
        self.label_train = y_train
        self.image_test = X_test
        self.label_test = y_test