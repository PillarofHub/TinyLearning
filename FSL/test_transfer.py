# import common library
import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt
import time
import sys

# import custom library
from SimilarNet import SimilarNet
from PinchLearning import PinchLearning
from OmniglotDataset import OmniglotDataset
from PairGenTrain import PairGenTrain
from PairGenTest import PairGenTest

# get params and dataset size
dataset_size_arr = [
    [5,1],
    [5,5],
    [5,10],
    [20,1],
    [20,5],
    [20,10],
    [964,10]
]

import sys
verbose_param = 0
if len(sys.argv) >= 3:
    if sys.argv[2].isdigit() is True:
        if int(sys.argv[2]) >= 0 and int(sys.argv[2]) < 3:
            verbose_param = int(sys.argv[2])

if len(sys.argv) < 2: sys.exit()
if sys.argv[1].isdigit() is False: sys.exit("dataset size indicator must be number")
if int(sys.argv[1]) < 0: sys.exit("dataset size indicator cannot be below zero")
if int(sys.argv[1]) >= len(dataset_size_arr): sys.exit("dataset size indicator cannot be over than dataset size arr")
dataset_size_indicator = int(sys.argv[1])

test_way_param = [5,20]
test_shot_param = [1,5]

# load raw dataset
datasetStr = "omniglot"

ds = OmniglotDataset()
(X_train_total, y_train_total) = (ds.image_train, ds.label_train)
(X_test_total, y_test_total) = (ds.image_test, ds.label_test)

# shrink and split dataset
class_labels = np.random.choice(range(np.min(y_test_total)), dataset_size_arr[dataset_size_indicator][0], replace=False)
num_classes = len(class_labels)
dataset_size = dataset_size_arr[dataset_size_indicator][1]

train_index_list = []
for label in class_labels:
    train_index_list.append(np.random.choice(np.where(y_train_total == label)[0], dataset_size * 2, replace=False))

X_support_list = []
y_support_list = []

for labelIndex in range(len(class_labels)):
    for index in train_index_list[labelIndex][:dataset_size]:
        X_support_list.append(X_train_total[index])
        y_support_list.append(y_train_total[index])

X_train_list = []
y_train_list = []

for labelIndex in range(len(class_labels)):
    for index in train_index_list[labelIndex][:dataset_size]:
        X_train_list.append(X_train_total[index])
        y_train_list.append(y_train_total[index])

X_valid_list = []
y_valid_list = []

for labelIndex in range(len(class_labels)):
    for index in train_index_list[labelIndex][dataset_size:]:
        X_valid_list.append(X_train_total[index])
        y_valid_list.append(y_train_total[index])

X_support = np.array(X_support_list)
y_support = np.array(y_support_list)
X_train = np.array(X_train_list)
y_train = np.array(y_train_list)
X_valid = np.array(X_valid_list)
y_valid = np.array(y_valid_list)

# create pair generator
pairgen = PairGenTrain(X_support, y_support, X_train, y_train, X_valid, y_valid)
ds_train = pairgen.ds_train
ds_valid = pairgen.ds_valid

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
train_epoches = 10000
train_patience = 100

# train a model
def trainer(trainingModel, modelTag):
    tf.keras.backend.clear_session()
    trainingModel.compile(loss="mse", optimizer="adam", metrics=['accuracy'])

    tensorboard_cb = tf.keras.callbacks.TensorBoard("./logs/run_" + modelTag + "_" + str(save_seq_code))
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=train_patience, restore_best_weights=True)
    history = trainingModel.fit(ds_train, validation_data=ds_valid, epochs=train_epoches, callbacks=[tensorboard_cb, early_stopping_cb], verbose=verbose_param)

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

# paper model is ommited because paper model cannot do few-shot learning

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

modelCheckArr = check_comparator_model()

## calculate 1:1 comparison accuracy
# make a test dataset

class_labels_test = np.random.choice(range(np.min(y_test_total), np.max(y_test_total)+1), 659, replace=False)
num_classes_test = len(class_labels_test)
dataset_size_test = dataset_size_arr[dataset_size_indicator]

test_index_list = []
for label in class_labels_test:
    test_index_list.append(np.random.choice(np.where(y_test_total == label)[0], 20, replace=False))

X_support_test_list = []
y_support_test_list = []

for labelIndex in range(len(class_labels_test)):
    for index in test_index_list[labelIndex][:dataset_size]:
        X_support_test_list.append(X_test_total[index])
        y_support_test_list.append(y_test_total[index])

X_test_list = []
y_test_list = []

for labelIndex in range(len(class_labels_test)):
    for index in test_index_list[labelIndex][:dataset_size]:
        X_test_list.append(X_test_total[index])
        y_test_list.append(y_test_total[index])

X_support_test = np.array(X_support_test_list)
y_support_test = np.array(y_support_test_list)
X_test = np.array(X_test_list)
y_test = np.array(y_test_list)

# create pair generator
pairgen = PairGenTest(X_support_test, y_support_test, X_test, y_test)
ds_test = pairgen.ds_test

# evaluate comparison accuracy
tf.keras.backend.clear_session()
accuracy_single_similarnet_cosine = whole_similarnet_cosine_model.evaluate(ds_test)[1]
accuracy_single_concat_cosine = whole_concat_cosine_model.evaluate(ds_test)[1]
accuracy_single_similarnet_compact = whole_similarnet_compact_model.evaluate(ds_test)[1]
accuracy_single_concat_compact = whole_concat_compact_model.evaluate(ds_test)[1]
accuracy_single_similarnet_full = whole_similarnet_full_model.evaluate(ds_test)[1]
accuracy_single_concat_full = whole_concat_full_model.evaluate(ds_test)[1]
tf.keras.backend.clear_session()


## make test datasets
# makes n-way k-shot support set and single test data
def episodeMaker(way_param, shot_param):
    if way_param is None: way_param = len(np.unique(y_test_total))
    class_labels = np.random.choice(range(np.min(y_test_total), np.max(y_test_total) + 1), way_param, replace=False)
    num_classes = len(class_labels)

    test_index_list = []
    for label in class_labels:
        test_index_list.append(np.random.choice(np.where(y_test_total == label)[0], shot_param + 1, replace=False))

    X_support_list = []
    y_support_list = []

    for labelIndex in range(len(class_labels)):
        for index in test_index_list[labelIndex][:shot_param]:
            X_support_list.append(X_test_total[index])
            y_support_list.append(y_test_total[index])

    X_test_list = []
    y_test_list = []

    class_labels_index_test = np.random.choice(len(class_labels), 1, replace=False)
    data_index_test = test_index_list[class_labels_index_test[0]][shot_param]
    X_test_list.append(X_test_total[data_index_test])
    y_test_list.append(y_test_total[data_index_test])

    X_support = np.array(X_support_list)
    y_support = np.array(y_support_list)
    X_test = np.array(X_test_list)
    y_test = np.array(y_test_list)

    return X_support, y_support, X_test, y_test

## classification
def calculateAccuracy(test_count=100):
    tf.keras.backend.clear_session()
    correct_support = np.zeros((4,6))
    wayList = [test_way_param[0], test_way_param[0], test_way_param[1], test_way_param[1]]
    shotList = [test_shot_param[0], test_shot_param[1], test_shot_param[0], test_shot_param[1]]

    for test_index in range(test_count):
        for wayshot_index in range(4):
            print(str(test_index+1) + " / " + str(test_count) + "            ", end = "\r")
            X_support, y_support, X_test, y_test = episodeMaker(wayList[wayshot_index], shotList[wayshot_index])

            pinch_similarnet_cosine = PinchLearning(whole_similarnet_cosine_model, X_support, y_support)
            pinch_concat_cosine = PinchLearning(whole_concat_cosine_model, X_support, y_support)
            pinch_similarnet_compact = PinchLearning(whole_similarnet_compact_model, X_support, y_support)
            pinch_concat_compact = PinchLearning(whole_concat_compact_model, X_support, y_support)
            pinch_similarnet_full = PinchLearning(whole_similarnet_full_model, X_support, y_support)
            pinch_concat_full = PinchLearning(whole_concat_full_model, X_support, y_support)

            testModelList = [pinch_similarnet_cosine, pinch_concat_cosine, pinch_similarnet_compact, pinch_concat_compact, pinch_similarnet_full, pinch_concat_full]

            for cnt in range(len(testModelList)):
                y_predict = testModelList[cnt].classify(X_test[0])
                y_truth = y_test[0]
                if y_predict == y_truth: correct_support[wayshot_index][cnt] += 1

    return correct_support / test_count

classifyAccuracyArr = calculateAccuracy(100)

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

    for append in appendix:
        file = open(dir_name + str(append) + ".tag", "w")
        file.close()

    np.save(dir_name + "X_train", X_train)
    np.save(dir_name + "X_valid", X_valid)

    import shutil
    shutil.move("./logs", dir_name)

train_way_num = dataset_size_arr[dataset_size_indicator][0]
train_shot_num = dataset_size_arr[dataset_size_indicator][1]
saveModels(str(train_way_num) + "_" + str(train_shot_num) + "_" + str(save_seq_code),
           ["single_similarnet_cosine_compact_full_" + str(round(accuracy_single_similarnet_cosine, 4)) + "_" + str(round(accuracy_single_similarnet_compact, 4)) + "_" + str(round(accuracy_single_similarnet_full, 4)),
            "single_concat_cosine_compact_full_" + str(round(accuracy_single_concat_cosine, 4)) + "_" + str(round(accuracy_single_concat_compact, 4)) + "_" + str(round(accuracy_single_concat_full, 4)),
            "dataset_" + datasetStr,
            "train_way_" + str(train_way_num),
            "train_shot_" + str(train_shot_num),
            "accuracy_5_1_" + str(round(classifyAccuracyArr[0][0], 4)) + "_" + str(round(classifyAccuracyArr[0][1], 4)) + "_" + str(round(classifyAccuracyArr[0][2], 4)) + "_" + str(round(classifyAccuracyArr[0][3], 4)) + "_" + str(round(classifyAccuracyArr[0][4], 4)) + "_" + str(round(classifyAccuracyArr[0][5], 4)),
            "accuracy_5_5_" + str(round(classifyAccuracyArr[1][0], 4)) + "_" + str(round(classifyAccuracyArr[1][1], 4)) + "_" + str(round(classifyAccuracyArr[1][2], 4)) + "_" + str(round(classifyAccuracyArr[1][3], 4)) + "_" + str(round(classifyAccuracyArr[1][4], 4)) + "_" + str(round(classifyAccuracyArr[1][5], 4)),
            "accuracy_20_1_" + str(round(classifyAccuracyArr[2][0], 4)) + "_" + str(round(classifyAccuracyArr[2][1], 4)) + "_" + str(round(classifyAccuracyArr[2][2], 4)) + "_" + str(round(classifyAccuracyArr[2][3], 4)) + "_" + str(round(classifyAccuracyArr[2][4], 4)) + "_" + str(round(classifyAccuracyArr[2][5], 4)),
            "accuracy_20_5_" + str(round(classifyAccuracyArr[3][0], 4)) + "_" + str(round(classifyAccuracyArr[3][1], 4)) + "_" + str(round(classifyAccuracyArr[3][2], 4)) + "_" + str(round(classifyAccuracyArr[3][3], 4)) + "_" + str(round(classifyAccuracyArr[3][4], 4)) + "_" + str(round(classifyAccuracyArr[3][5], 4)),
            "determine_similarnet_cosine_compact_full_" + str(determineArr[0]) + "_" + str(determineArr[1]) + "_" + str(determineArr[2]),
            "determine_concat_cosine_compact_full_" + str(determineArr[3]) + "_" + str(determineArr[4]) + "_" + str(determineArr[5])
            ]
           )

tf.keras.backend.clear_session()