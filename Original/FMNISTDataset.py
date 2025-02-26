class FMNISTDataset:
    def __init__(self):
        import tensorflow as tf
        import numpy as np

        (X_train, y_train), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

        X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
        X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

        X_train = X_train.astype('float32') / 255;
        X_test = X_test.astype('float32') / 255;

        self.image_train = X_train
        self.label_train = y_train
        self.image_test = X_test
        self.label_test = y_test