from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
import numpy as np

from Filter import *


class Harf:
    def __init__(self, nfolds = 10, agreementLevel = 0.7,
                 ntrees = 500, seed = 0):
        # Some data verification
        # Data can be a DataFrame or a Numpy Array
        if (agreementLevel < 0.5 or agreementLevel > 1):
            raise ValueError("Agreement Level must be between 0.5 and 1.")
        # if (classColumn < 0 or classColumn > len(data)):
        #     raise ValueError("Column of class out of data bounds")
        self.nfolds = nfolds
        self.agreementLevel = agreementLevel
        self.ntrees = ntrees
        self.seed = seed
        self.k_fold = KFold(nfolds, shuffle = True, random_state = self.seed)
        self.clf = RandomForestClassifier(n_estimators = ntrees, random_state = seed)

    def __call__(self, data, classes):
        self.splits = self.k_fold.split(data)
        self.isNoise = np.array([False] * len(classes))
        filter = Filter(parameters = {"nfolds": self.nfolds, "ntrees": self.ntrees, "agreementLevel": self.agreementLevel})
        for train_indx, test_indx in self.splits:
            self.clf.fit(data[train_indx], classes[train_indx])
            probs = self.clf.predict_proba(data[test_indx])
            self.isNoise[test_indx] = [prob[class_indx] <= 1 - self.agreementLevel
                                       for prob, class_indx in zip(probs, classes[test_indx])]
        filter.remIndx = np.argwhere(self.isNoise)
        notNoise = np.invert(self.isNoise)
        filter.set_cleanData(data[notNoise], classes[notNoise])
        return filter
