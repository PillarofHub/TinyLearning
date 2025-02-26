# PinchLearning class (keras 2)
# author : Yi Chungheon ( pillarofcode@gmail.com )

# import
from collections import defaultdict
import numpy as np

class PinchLearning:
    # model definition
    def __init__(self, comparator, X_support, y_support):
        self.comparator = comparator
        self.X_support = np.array(X_support)
        self.y_support = np.array(y_support)
        self.label_arr = np.unique(y_support)

        support_idx = defaultdict(list)
        for y_support_idx, y in enumerate(y_support):
            support_idx[y].append(y_support_idx)

        self.support_data_list = []
        for cnt in range(len(self.label_arr)):
            indexArr = support_idx[self.label_arr[cnt]]
            singleSupportDataList = []
            for index in indexArr:
                singleSupportDataList.append(X_support[index])
                singleSupportDataArr = np.array(singleSupportDataList)

            self.support_data_list.append(singleSupportDataArr)

    def print_raw(self, target):
        similarityResultList = []
        for cnt in range(len(self.label_arr)):
            support = self.support_data_list[cnt]
            targetBroadcasted = np.broadcast_to(target, (len(support), *np.shape(target)))
            similarityResult = self.comparator.predict((targetBroadcasted, support))
            similarityResultList.append(similarityResult)
        return similarityResultList

    def print_similarity(self, target):
        simRaw = self.print_raw(target)
        simList = []
        for cnt in range(len(simRaw)):
            meanValue = np.mean(simRaw[cnt])
            simList.append(meanValue)
        return np.array(simList)

    def classify(self,target):
        index = np.argmax(self.print_similarity(target))
        return self.label_arr[index]
    
    def evaluate(self,target,truth):
        simArr = self.print_similarity(target)
        truth_index = np.where(self.label_arr == truth)[0][0]
        rankArr = np.argsort(-simArr)
        rank = np.where(rankArr == truth_index)[0][0] + 1
        return rank
        