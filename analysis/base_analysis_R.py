import os
import pandas as pd
import numpy as np

from sklearn import svm, linear_model, discriminant_analysis, neighbors
from sklearn import tree, naive_bayes, ensemble, neural_network, gaussian_process
from sklearn.model_selection import cross_validate, KFold
from sklearn import metrics
from sklearn import preprocessing

import constants
from Default import Default
from Random import Random
from meta_db.db.DBHelper import DBHelper

from R_Model import *
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

# From Pandas to R
# with localconverter(ro.default_converter + pandas2ri.converter):
#  r_from_pd_df = ro.conversion.py2rpy(pd_df)

# From R to Pandas
# with localconverter(ro.default_converter + pandas2ri.converter):
#   pd_from_r_df = ro.conversion.rpy2py(r_df)

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

# # Test
from sklearn.datasets import load_iris
iris = load_iris()
feature_names = [name.replace("(cm)", " ").strip().replace(" ", "_") for name in iris.feature_names]
iris_df = pd.DataFrame(iris.data, columns = feature_names)
iris_df["class"] = iris.target
train_iris = iris_df[:120]
test_iris = iris_df[120:]
with localconverter(ro.default_converter + pandas2ri.converter):
    train_iris_R = r_from_pd_df = ro.conversion.py2rpy(train_iris)
    test_iris_R = r_from_pd_df = ro.conversion.py2rpy(test_iris)

formula = Formula("class ~" + "+ ".join(feature_names))

nn = neuralnet.neuralnet(formula, data = train_iris_R)
result = np.array(ro.r.predict(nn, test_iris_R))

#____________________________________________________________#

SCORE = "balanced_accuracy_mean"
REPETITIONS = 30

translator = {
    "svm": "SVM",
    "logistic_regression": "LG",
    "linear_discriminant": "LD",
    "kneighbors": "kNN",
    "decision_tree": "DT",
    "gaussian_nb": "GNB",
    "random_forest": "RF",
    "randomForest": "RF",
    "gradient_boosting": "GB",
    "neural_network": "NN",
    "knn": "kNN",
    "dwnn": "DWNN",
    "Svm": "SVM",
    "ann": "ANN",
    "cart": "CART",
    "random": "Random",
    "default": "Default"
}

db = DBHelper()

if not os.path.exists("analysis/plots"):
    os.makedirs("analysis/plots")
if not os.path.exists("analysis/plots/base_analysis"):
    os.makedirs("analysis/plots/base_analysis")


metadata = pd.DataFrame(db.get_all_metadata(), columns = db.metadata_columns()).drop("id", axis = 1)
models = pd.DataFrame(db.get_all_models(), columns = db.models_columns()).drop("id", axis = 1)
combinations = pd.DataFrame(db.get_all_combinations(), columns = db.combinations_columns())
preperformance = pd.DataFrame(db.get_all_preperformance(), columns = db.preperformance_columns()).drop("id", axis = 1)
# Not null preperformance
preperformance = preperformance[~preperformance.isnull().any(axis = 1)]
preperformance = pd.merge(preperformance, combinations, left_on = "combination_id", right_on = "id").drop(["combination_id", "id", "num_preprocesses"], axis = 1)

models = models.rename(columns = {"model": "classifier"})
models["preprocesses"] = "None"
scores = pd.concat([models, preperformance], sort = False)
scores = scores[scores.preprocesses.isin(constants.PRE_PROCESSES + ["None"]) & scores.classifier.isin(constants.CLASSIFIERS)]
scores = scores.drop(columns = ["recall_micro_mean", "recall_micro_std", "precision_macro_mean", "precision_macro_std", "recall_macro_std", "recall_macro_mean", "accuracy_mean", "accuracy_std","precision_micro_mean", "precision_micro_std", "f1_micro_mean", "f1_micro_std", "f1_macro_std", "f1_macro_mean","fit_time_mean", "fit_time_std", "score_time_mean", "score_time_std" ])

metadata_means = {feature: np.mean(metadata[feature]) for feature in metadata.columns if feature != "name"}
metadata.fillna(value = metadata_means, inplace = True)

data = pd.merge(metadata, scores, on = "name")

data = data[data.preprocesses.isin(constants.PRE_PROCESSES + ["None"]) & data.classifier.isin(constants.CLASSIFIERS)]
combinations_strings = ["{}+{}".format(pp, clf) for pp in ["None"] + constants.PRE_PROCESSES
                                                for clf in constants.CLASSIFIERS]

mean_scores = []
std_scores = []
for score in constants.CLASSIFIERS_SCORES:
    mean_scores.append(score + "_mean")
    std_scores.append(score + "_std")

# Function to get only datasets with all results (combinations)
def filter_dataset(database):
    datasets_filtered = []
    for dataset in database.name.unique():
        split = database[database.name == dataset]
        keep = True
        for clf in constants.CLASSIFIERS:
            for pp in constants.PRE_PROCESSES + ['None']:
                if len(split[(split.classifier == clf) & (split.preprocesses == pp)]) < 1:
                    keep = False
        if keep:
            datasets_filtered.append(dataset)
    return datasets_filtered

datasets = pd.Series(filter_dataset(data))
print("Num datasets:", len(datasets))

# Redimensioning the metabase in data (removing other scores)
list_datasets = []
for dataset in datasets:
    meta_features = metadata[metadata.name == dataset]
    dataset_dict = {col:value for col, value in zip(list(meta_features), list(meta_features.values[0]))}
    for pp in ['None'] + constants.PRE_PROCESSES:
        for clf in constants.CLASSIFIERS:
            dataset_dict["{}+{}".format(pp, clf)] = float(scores[(scores.name == dataset) & (scores.preprocesses == pp) & (scores.classifier == clf)]["balanced_accuracy_mean"])
    list_datasets.append(dataset_dict)

meta_base = pd.DataFrame(list_datasets)
for rep in range(REPETITIONS):
    rand_state = np.random.randint(low = 1, high = 1000)
    np.random.seed(rand_state)
    divideFold = KFold(10, random_state = rand_state, shuffle = True)

    reg_models = {}
    reg_models["ann"] = lambda: R_Model(neuralnet.neuralnet)
    reg_models["cart"] = lambda: R_Model(rpart.rpart)
    reg_models["randomForest"] = lambda: R_Model(randomForest.randomForest)
    reg_models["dwnn"] = lambda: KNN()
    reg_models["random"] = lambda: Random(random_seed = rand_state)
    reg_models["default"] = lambda: Default()

    predictions = []
    # Divide datasets in train and test
    for train_indx, test_indx in divideFold.split(datasets):
        models = {}
        # Selecting only data for test
        tests = meta_base[meta_base.name.isin(list(datasets.iloc[test_indx]))]
        # Selecting only data from train
        train_data = meta_base[meta_base.name.isin(list(datasets.iloc[train_indx]))]
        targets = train_data.drop(combinations_strings + ["name"], axis = 1)
        # Training block
        for comb in combinations_strings:
            models[comb] = {}
            labels = train_data[comb].values
            for regressor_type in constants.REGRESSORS:
                models[comb][regressor_type] = reg_models[regressor_type]()
                models[comb][regressor_type].fit(targets, labels)
        # Test block
        for test_dataset in tests.name:
            to_predict_input = meta_base[meta_base.name == test_dataset].drop(
                combinations_strings + ["name"], axis = 1
            )
            for regressor_type in constants.REGRESSORS:
                prediction = {"dataset": test_dataset, "regressor": regressor_type}
                prediction.update({comb: float(models[comb][regressor_type].predict(to_predict_input))
                                 for comb in combinations_strings})
                predictions.append(prediction)
    predictions = pd.DataFrame(predictions,
                       columns = ["dataset", "regressor", *combinations_strings])
    predictions.to_csv("analysis/plots/base_analysis/predictions_{}.csv".format(rep), sep = ",")
