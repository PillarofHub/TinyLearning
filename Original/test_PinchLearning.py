# import common library
import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt
import time

# import custom library
from SimilarNet import SimilarNet
from PinchLearning import PinchLearning
from MNISTDataset import MNISTDataset
from FMNISTDataset import FMNISTDataset
from CIFARDataset import CIFARDataset
from SVHNDataset import SVHNDataset
from PairGenS import PairGenS

# get params and dataset size
#dataset_size_arr = [1,2,3,4,5,8,10,15,16,20,30,32,40,50,60,64]
#dataset_size_arr = [1,2,4,8, 10, 20, 40, 80]
dataset_size_arr = [8]

import sys
verbose_param = 0
if len(sys.argv) >= 4:
    if sys.argv[3].isdigit() is True:
        if int(sys.argv[3]) >= 0 and int(sys.argv[3]) < 3:
            verbose_param = int(sys.argv[3])

if len(sys.argv) < 3: sys.exit()
if sys.argv[2].isdigit() is False: sys.exit("dataset size indicator must be number")
if int(sys.argv[2]) < 0: sys.exit("dataset size indicator cannot be below zero")
if int(sys.argv[2]) >= len(dataset_size_arr): sys.exit("dataset size indicator cannot be over than dataset size arr")
dataset_size_indicator = int(sys.argv[2])

# load raw dataset
datasetStr = sys.argv[1].lower()

ds = None
if datasetStr == "cifar" : ds = CIFARDataset()
elif datasetStr == "fmnist" : ds = FMNISTDataset()
elif datasetStr == "svhn" : ds = SVHNDataset()
else: ds = MNISTDataset()
(X_train_total, y_train_total) = (ds.image_train, ds.label_train)
(X_test_total, y_test_total) = (ds.image_test, ds.label_test)

# shrink and split dataset
class_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
num_classes = len(class_labels)
dataset_size = dataset_size_arr[dataset_size_indicator]

train_index_list = []
for label in class_labels:
    train_index_list.append(np.random.choice(np.where(y_train_total == label)[0], dataset_size * 2, replace=False))

X_support_list = []
y_support_list = []

for label in class_labels:
    for index in train_index_list[label][:dataset_size]:
        X_support_list.append(X_train_total[index])
        y_support_list.append(y_train_total[index])

X_train_list = []
y_train_list = []

for label in class_labels:
    for index in train_index_list[label][:dataset_size]:
        X_train_list.append(X_train_total[index])
        y_train_list.append(y_train_total[index])

X_valid_list = []
y_valid_list = []

for label in class_labels:
    for index in train_index_list[label][dataset_size:]:
        X_valid_list.append(X_train_total[index])
        y_valid_list.append(y_train_total[index])

test_index_list = []
for label in class_labels:
    test_index_list.append(np.random.choice(np.where(y_test_total == label)[0], 10, replace=False))

X_test_list = []
y_test_list = []

for label in class_labels:
    for index in test_index_list[label]:
        X_test_list.append(X_test_total[index])
        y_test_list.append(y_test_total[index])

X_support = np.array(X_support_list)
y_support = np.array(y_support_list)
X_train = np.array(X_train_list)
y_train = np.array(y_train_list)
X_valid = np.array(X_valid_list)
y_valid = np.array(y_valid_list)
X_test = np.array(X_test_list)
y_test = np.array(y_test_list)

# create pair generator
pairgen = PairGenS(X_support, y_support, X_train, y_train, X_valid, y_valid, X_test, y_test)
ds_train = pairgen.ds_train
ds_valid = pairgen.ds_valid
ds_test = pairgen.ds_test

## model definition
# embedding model function
def embeddingModelBuilder():
    retModel = tf.keras.models.Sequential()
    retModel.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), padding="valid"))
    retModel.add(tf.keras.layers.BatchNormalization())
    retModel.add(tf.keras.layers.ReLU())
    retModel.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))

    retModel.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), padding="valid"))
    retModel.add(tf.keras.layers.BatchNormalization())
    retModel.add(tf.keras.layers.ReLU())
    retModel.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))

    retModel.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), padding="same"))
    retModel.add(tf.keras.layers.BatchNormalization())
    retModel.add(tf.keras.layers.ReLU())

    retModel.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), padding="same"))
    retModel.add(tf.keras.layers.BatchNormalization())
    retModel.add(tf.keras.layers.ReLU())

    return retModel

# comparator model function
def comparatorModelBuilder(isFullModel=True) :
    retModel = tf.keras.models.Sequential()
    if isFullModel:
        retModel.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding="same"))
        retModel.add(tf.keras.layers.BatchNormalization())
        retModel.add(tf.keras.layers.ReLU())
        retModel.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

        retModel.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding="same"))
        retModel.add(tf.keras.layers.BatchNormalization())
        retModel.add(tf.keras.layers.ReLU())
        retModel.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

    retModel.add(tf.keras.layers.Flatten())
    retModel.add(tf.keras.layers.Dense(8, activation="relu"))
    retModel.add(tf.keras.layers.Dense(1, activation="sigmoid"))

    return retModel

embedding_similarnet_full_model = embeddingModelBuilder()
comparator_similarnet_full_model = comparatorModelBuilder(isFullModel = True)

embedding_similarnet_compact_model = embeddingModelBuilder()
comparator_similarnet_compact_model = comparatorModelBuilder(isFullModel = False)

embedding_concat_full_model = embeddingModelBuilder()
comparator_concat_full_model = comparatorModelBuilder(isFullModel = True)

embedding_concat_compact_model = embeddingModelBuilder()
comparator_concat_compact_model = comparatorModelBuilder(isFullModel = False)

embedding_similarnet_cosine_model = embeddingModelBuilder()

embedding_concat_cosine_model = embeddingModelBuilder()

# paper model
paper_model = tf.keras.models.Sequential()
paper_model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding="valid"))
paper_model.add(tf.keras.layers.MaxPooling2D((2, 2)))
paper_model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding="valid"))
paper_model.add(tf.keras.layers.MaxPooling2D((2, 2)))
paper_model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding="same"))
paper_model.add(tf.keras.layers.MaxPooling2D((2, 2)))
paper_model.add(tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding="same"))
paper_model.add(tf.keras.layers.MaxPooling2D((2, 2)))
paper_model.add(tf.keras.layers.Flatten())
paper_model.add(tf.keras.layers.Dropout(0.7))
paper_model.add(tf.keras.layers.Dense(num_classes, activation="sigmoid"))

## architecture definition
# metric space learning model
class MetricSpaceLearning(tf.keras.Model):
    def __init__(self, embeddingModel=tf.keras.layers.Lambda(lambda x: x),
                 comparatorModel=tf.keras.layers.Lambda(lambda x: x),
                 isSimilarNet=True, **kwargs):
        super().__init__(**kwargs)

        self.embedding = embeddingModel
        self.similarnet = SimilarNet()
        if isSimilarNet == False: self.similarnet = tf.keras.layers.Concatenate()
        self.comparator = comparatorModel

    def call(self, inputs):
        input_A, input_B = inputs[0], inputs[1]
        tensor_A = self.embedding(input_A)
        tensor_B = self.embedding(input_B)

        tensor = self.similarnet((tensor_A, tensor_B))

        tensor = self.comparator(tensor)
        return tensor

# cosine based siamese learning model
class CosineLearning(tf.keras.Model):
    def __init__(self, embeddingModel=tf.keras.layers.Lambda(lambda x: x), isSimilarNet=True, **kwargs):
        super().__init__(**kwargs)

        self.embedding = embeddingModel
        self.similarnet = SimilarNet()
        if isSimilarNet == False: self.similarnet = tf.keras.layers.Concatenate()
        self.flatten = tf.keras.layers.Flatten()

    def call(self, inputs):
        input_A, input_B = inputs[0], inputs[1]
        tensor_A = self.embedding(input_A)
        tensor_B = self.embedding(input_B)

        tensor = self.similarnet((tensor_A, tensor_B))
        tensor = self.flatten(tensor)

        tensor = tf.math.reduce_sum(tensor, axis=-1, keepdims=True)
        return tensor


save_seq_code = time.strftime("%y%m%d-%H%M%S")

# train a model
def trainer(trainingModel, modelTag):
    tf.keras.backend.clear_session()
    trainingModel.compile(loss="mse", optimizer="adam", metrics=['accuracy'])

    tensorboard_cb = tf.keras.callbacks.TensorBoard("./logs/run_" + modelTag + "_" + str(save_seq_code))
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=100, restore_best_weights=True)
    history = trainingModel.fit(ds_train, validation_data=ds_valid, epochs=10000, callbacks=[tensorboard_cb, early_stopping_cb], verbose=verbose_param)

whole_similarnet_cosine_model = CosineLearning(embeddingModel=embedding_similarnet_cosine_model, isSimilarNet=True)
trainer(whole_similarnet_cosine_model, "similarnet_cosine")

whole_concat_cosine_model = CosineLearning(embeddingModel=embedding_concat_cosine_model, isSimilarNet=False)
trainer(whole_concat_cosine_model, "concat_cosine")

whole_similarnet_compact_model = MetricSpaceLearning(embeddingModel=embedding_similarnet_compact_model, comparatorModel=comparator_similarnet_compact_model, isSimilarNet=True)
trainer(whole_similarnet_compact_model, "similarnet_compact")

whole_concat_compact_model = MetricSpaceLearning(embeddingModel=embedding_concat_compact_model, comparatorModel=comparator_concat_compact_model, isSimilarNet=False)
trainer(whole_concat_compact_model, "concat_compact")

whole_similarnet_full_model = MetricSpaceLearning(embeddingModel=embedding_similarnet_full_model, comparatorModel=comparator_similarnet_full_model, isSimilarNet=True)
trainer(whole_similarnet_full_model, "similarnet_full")

whole_concat_full_model = MetricSpaceLearning(embeddingModel=embedding_concat_full_model, comparatorModel=comparator_concat_full_model, isSimilarNet=False)
trainer(whole_concat_full_model, "concat_full")

# train a paper model
tf.keras.backend.clear_session()
y_train_onehot = tf.keras.utils.to_categorical(y_train, num_classes)
y_valid_onehot = tf.keras.utils.to_categorical(y_valid, num_classes)
y_test_onehot = tf.keras.utils.to_categorical(y_test, num_classes)

tf.keras.backend.clear_session()
paper_model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])

tensorboard_cb = tf.keras.callbacks.TensorBoard("./logs/run_paper_" + str(save_seq_code))
early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=100, restore_best_weights=True)
history = paper_model.fit(X_train, y_train_onehot, validation_data=(X_valid, y_valid_onehot), epochs=10000, callbacks = [tensorboard_cb, early_stopping_cb], verbose=verbose_param)

# calculate 1:1 comparison accuracy
tf.keras.backend.clear_session()
accuracy_single_similarnet_cosine = whole_similarnet_cosine_model.evaluate(ds_test)[1]
accuracy_single_concat_cosine = whole_concat_cosine_model.evaluate(ds_test)[1]
accuracy_single_similarnet_compact = whole_similarnet_compact_model.evaluate(ds_test)[1]
accuracy_single_concat_compact = whole_concat_compact_model.evaluate(ds_test)[1]
accuracy_single_similarnet_full = whole_similarnet_full_model.evaluate(ds_test)[1]
accuracy_single_concat_full = whole_concat_full_model.evaluate(ds_test)[1]
tf.keras.backend.clear_session()
accuracy_paper = paper_model.evaluate(X_test, y_test_onehot)[1]
tf.keras.backend.clear_session()

## Evaluate pinchlearning classifier accuracy
pinch_similarnet_cosine = PinchLearning(whole_similarnet_cosine_model, X_support, y_support)
pinch_concat_cosine = PinchLearning(whole_concat_cosine_model, X_support, y_support)
pinch_similarnet_compact = PinchLearning(whole_similarnet_compact_model, X_support, y_support)
pinch_concat_compact = PinchLearning(whole_concat_compact_model, X_support, y_support)
pinch_similarnet_full = PinchLearning(whole_similarnet_full_model, X_support, y_support)
pinch_concat_full = PinchLearning(whole_concat_full_model, X_support, y_support)

## classification
def calculateAccuracy(testModel):
    tf.keras.backend.clear_session()
    correct_support = 0

    for test_index in range(len(y_test)):
        print(str(test_index+1) + " / " + str(len(y_test)) + "            ", end = "\r")
        y_predict = testModel.classify(X_test[test_index])
        y_truth = y_test[test_index]
        if y_predict == y_truth: correct_support += 1

    accuracy_comp_class = correct_support / len(y_test)

    return accuracy_comp_class

accuracy_similarnet_cosine = calculateAccuracy(pinch_similarnet_cosine)
accuracy_concat_cosine = calculateAccuracy(pinch_concat_cosine)
accuracy_similarnet_compact = calculateAccuracy(pinch_similarnet_compact)
accuracy_concat_compact = calculateAccuracy(pinch_concat_compact)
accuracy_similarnet_full = calculateAccuracy(pinch_similarnet_full)
accuracy_concat_full = calculateAccuracy(pinch_concat_full)

# check determinability of comparotor model
def check_comparator_model():
    def make_data():
        dataset = ds_train

        X_train_comp_left_list = []
        X_train_comp_right_list = []
        y_train_comp_list = []

        for batch in dataset.repeat().take(8):
            for icnt in range(32):
                X_train_comp_left_list.append(batch[0][0][icnt])
                X_train_comp_right_list.append(batch[0][1][icnt])
                y_train_comp_list.append(batch[1][icnt][0])

        X_train_comp_left = tf.convert_to_tensor(np.array(X_train_comp_left_list))
        X_train_comp_right = tf.convert_to_tensor(np.array(X_train_comp_right_list))
        y_train_comp = tf.convert_to_tensor(np.array(y_train_comp_list))

        X_train_comp = np.array([X_train_comp_left, X_train_comp_right])

        return X_train_comp, y_train_comp

    def check_model(model, X_train_comp, y_train_comp):
        y_pred_comp = np.squeeze(model.predict(X_train_comp))
        pred_pos = np.size(np.where(y_pred_comp > 0.5))
        pred_neg = np.size(np.where(y_pred_comp <= 0.5))

        return 1 if (pred_pos != 0 and pred_neg != 0) else 0

    X_train_comp = None
    y_train_comp = None
    making_data = True
    while y_train_comp == None:
        X_train_comp, y_train_comp = make_data()
        truth_pos = np.size(np.where(y_train_comp > 0.5))
        truth_neg = np.size(np.where(y_train_comp <= 0.5))
        if truth_pos == 0 or truth_neg == 0:
            y_train_comp = None

    resultArr = []
    modelArr = [whole_similarnet_cosine_model, whole_similarnet_compact_model, whole_similarnet_full_model, whole_concat_cosine_model, whole_concat_compact_model, whole_concat_full_model]
    for model in modelArr:
        resultArr.append(check_model(model, X_train_comp, y_train_comp))

    return resultArr

determineArr = check_comparator_model()

## save model and results
def saveModels(tag, appendix=None):
    dir_name = "./models/" + datasetStr + "_" + str(tag) + "/"
    embedding_similarnet_full_model.save(dir_name + "embedding_similarnet_full.tfmodel")
    comparator_similarnet_full_model.save(dir_name + "comparator_similarnet_full.tfmodel")
    whole_similarnet_full_model.save(dir_name + "whole_similarnet_full.tfmodel")

    embedding_concat_full_model.save(dir_name + "embedding_concat_full.tfmodel")
    comparator_concat_full_model.save(dir_name + "comparator_concat_full.tfmodel")
    whole_concat_full_model.save(dir_name + "whole_concat_full.tfmodel")

    embedding_similarnet_compact_model.save(dir_name + "embedding_similarnet_compact.tfmodel")
    comparator_similarnet_compact_model.save(dir_name + "comparator_similarnet_compact.tfmodel")
    whole_similarnet_compact_model.save(dir_name + "whole_similarnet_compact.tfmodel")

    embedding_concat_compact_model.save(dir_name + "embedding_concat_compact.tfmodel")
    comparator_concat_compact_model.save(dir_name + "comparator_concat_compact.tfmodel")
    whole_concat_compact_model.save(dir_name + "whole_concat_compact.tfmodel")

    embedding_similarnet_cosine_model.save(dir_name + "embedding_similarnet_cosine.tfmodel")
    whole_similarnet_cosine_model.save(dir_name + "whole_similarnet_cosine.tfmodel")

    embedding_concat_cosine_model.save(dir_name + "embedding_concat_cosine.tfmodel")
    whole_concat_cosine_model.save(dir_name + "whole_concat_cosine.tfmodel")

    paper_model.save(dir_name + "paper.tfmodel")

    for append in appendix:
        file = open(dir_name + str(append) + ".tag", "w")
        file.close()

    np.save(dir_name + "X_train", X_train)
    np.save(dir_name + "X_valid", X_valid)
    np.save(dir_name + "X_test", X_test)

    np.save(dir_name + "y_train", y_train)
    np.save(dir_name + "y_valid", y_valid)
    np.save(dir_name + "y_test", y_test)

    import shutil
    shutil.move("./logs", dir_name)

saveModels(str(dataset_size) + "_" + str(save_seq_code),
           ["single_similarnet_cosine_compact_full_" + str(round(accuracy_single_similarnet_cosine, 4)) + "_" + str(round(accuracy_single_similarnet_compact, 4)) + "_" + str(round(accuracy_single_similarnet_full, 4)),
            "single_concat_cosine_compact_full_" + str(round(accuracy_single_concat_cosine, 4)) + "_" + str(round(accuracy_single_concat_compact, 4)) + "_" + str(round(accuracy_single_concat_full, 4)),
            "dataset_" + datasetStr,
            "numclass_" + str(num_classes),
            "size_" + str(dataset_size),
            "accuracy_similarnet_cosine_compact_full_" + str(round(accuracy_similarnet_cosine, 4)) + "_" + str(round(accuracy_similarnet_compact, 4)) + "_" + str(round(accuracy_similarnet_full, 4)),
            "accuracy_concat_cosine_compact_full_" + str(round(accuracy_concat_cosine, 4)) + "_" + str(round(accuracy_concat_compact, 4)) + "_" + str(round(accuracy_concat_full, 4)),
            "accuracy_paper_" + str(round(accuracy_paper, 4)),
            "determine_similarnet_cosine_compact_full_" + str(determineArr[0]) + "_" + str(determineArr[1]) + "_" + str(determineArr[2]),
            "determine_concat_cosine_compact_full_" + str(determineArr[3]) + "_" + str(determineArr[4]) + "_" + str(determineArr[5])
            ]
           )

tf.keras.backend.clear_session()