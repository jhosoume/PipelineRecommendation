import numpy as np
import pandas as pd

# Importing utils from R
from rpy2.robjects.packages import importr
import rpy2.robjects.packages as rpackages
from rpy2.robjects.vectors import StrVector
from rpy2.robjects.packages import SignatureTranslatedAnonymousPackage

# For Formulae
from rpy2.robjects import IntVector, Formula

# For Pandas
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects import numpy2ri
from rpy2.robjects.conversion import localconverter

# Getting R functions
utils = importr("utils")
utils.chooseCRANmirror(ind=1)

# Geting packages
packages = ("reshape2", "e1071", "kknn", "randomForest", "C50", "rpart", "neuralnet")
to_install = [pack for pack in packages if not rpackages.isinstalled(pack)]
if to_install:
    utils.install_packages(StrVector(to_install))

e1071 = importr("e1071")
kknn = importr("kknn")
randomForest = importr("randomForest")
c50 = importr("C50")
rpart = importr("rpart")
neuralnet = importr("neuralnet")

#___________________________________________________________#

class R_Model:
    def __init__(self, r_model):
        self.r_model = r_model

    def fit(self, train_data, labels, formula = "class ~ .", feature_names = ""):
        train_data = pd.DataFrame(train_data)
        # train should be a dataframe and labels a numpy.array
        train_data["class"] = labels
        with localconverter(ro.default_converter + pandas2ri.converter):
            train_R = r_from_pd_df = ro.conversion.py2rpy(train_data)
        if type(formula) == type("string"):
            if feature_names:
                formula = Formula("class ~" + "+ ".join(feature_names))
            else:
                formula = Formula(formula)
        self.trained = self.r_model(formula, data = train_R)
        return self.trained

    def predict(self, test_data):
        with localconverter(ro.default_converter + pandas2ri.converter):
            test_R = ro.conversion.py2rpy(test_data)
        return np.array(ro.r.predict(self.trained, test_R))


class KNN(R_Model):

    def __init__(self):
        R_Model.__init__(self, kknn.kknn)

    def fit(self, train_data, labels, formula = "class ~ .", feature_names = "", test_data = None, kernel = "gaussian"):
        # train should be a dataframe and labels a numpy.array
        train_data = pd.DataFrame(train_data)
        train_data["class"] = labels
        with localconverter(ro.default_converter + pandas2ri.converter):
            self.train_R = r_from_pd_df = ro.conversion.py2rpy(train_data)
        if type(formula) == type("string"):
            if feature_names:
                formula = Formula("class ~" + "+ ".join(feature_names))
            else:
                formula = Formula(formula)
        self.formula = formula
        self.kernel = kernel
        return self.r_model

    def predict(self, test_data):
        with localconverter(ro.default_converter + pandas2ri.converter):
            test_R = ro.conversion.py2rpy(test_data)
        res = self.r_model(self.formula, self.train_R, test_R, kernel = self.kernel)
        return np.array(res[0])

class SVR(R_Model):
    def __init__(self):
        R_Model.__init__(self, e1071.svm)

    def fit(self, train_data, labels, formula = "class ~ .", feature_names = ""):
        # train should be a dataframe and labels a numpy.array
        train_data = pd.DataFrame(train_data)
        train_data["class"] = labels
        with localconverter(ro.default_converter + pandas2ri.converter):
            train_R = r_from_pd_df = ro.conversion.py2rpy(train_data)
        if type(formula) == type("string"):
            if feature_names:
                formula = Formula("class ~" + "+ ".join(feature_names))
            else:
                formula = Formula(formula)
        self.trained = self.r_model(formula, data = train_R, scale = True, type = "eps-regression", kernel = "radial")
        return self.trained

# class ModelCollection:
#     def __init__(self):
#         self.models = {}
#         for name, model in zip(
#             ["ANN", "CART", "RF"],
#             [neuralnet.neuralnet, rpart.rpart, randomForest.randomForest]):
#             models[name] = R_Model(model)
