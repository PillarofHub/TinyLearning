class SVHNDataset:
    def __init__(self):
        import tensorflow as tf
        import tensorflow_datasets as tfds
        import numpy as np

        (ds_train, ds_test), ds_info = tfds.load(
            'svhn_cropped',
            split=['train', 'test'],
            as_supervised=True,
            with_info=True
        )

        imageList_train = []
        labelList_train = []
        imageList_test = []
        labelList_test = []

        def preprocess(image, label):
            image = tf.cast(image, tf.float32) / 255.0
            return image, label

        for image, label in ds_train.map(preprocess):
            image = image.numpy()
            label = label.numpy()

            imageList_train.append(image)
            labelList_train.append(label)

        for image, label in ds_test.map(preprocess):
            image = image.numpy()
            label = label.numpy()

            imageList_test.append(image)
            labelList_test.append(label)

        X_train = np.array(imageList_train)
        y_train = np.array(labelList_train)
        X_test = np.array(imageList_test)
        y_test = np.array(labelList_test)

        self.image_train = X_train
        self.label_train = y_train
        self.image_test = X_test
        self.label_test = y_test