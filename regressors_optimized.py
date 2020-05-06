import sys
import os
import pandas as pd
import numpy as np
import pickle

from sklearn import svm, linear_model, discriminant_analysis, neighbors
from sklearn import tree, naive_bayes, ensemble, neural_network, gaussian_process
from sklearn.model_selection import cross_validate, KFold, GridSearchCV
from sklearn import metrics
from sklearn import preprocessing
from sklearn.utils import check_array

import constants
from Default import Default
from Random import Random
from meta_db.db.DBHelper import DBHelper

db = DBHelper()

SCORE_COLUMNS = ["name"] + constants.CLASSIFIERS

SCORES = ["max_error", "mean_absolute_error", "r2_score", "median_absolute_error", "mean_squared_error"]

CLF_SCORE = "f1_macro"

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
scores = scores[scores.classifier != "neural_network"]
models = models[models.classifier != "neural_network"]

metadata_means = {feature: np.mean(metadata[feature]) for feature in metadata.columns if feature != "name"}
metadata.fillna(value = metadata_means, inplace = True)

data = pd.merge(metadata, scores, on = "name")

reg_models = {}

# reg_models["neural_network"] = {}
# reg_models["neural_network"]["model"] = neural_network.MLPRegressor()
# reg_models["neural_network"]["params"] = {
#     "hidden_layer_sizes": [50, 100, 150],
#     "solver": ["sgd", "adam"]
# }
#

# reg_models["ridge"] = {}
# reg_models["ridge"]["model"] = linear_model.Ridge()
# reg_models["ridge"]["params"] = {
#     "alpha": [0.5, 1.0, 1.5],
#     "solver": ["svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga"]
# }
#
# reg_models["gradient_descent"] = {}
# reg_models["gradient_descent"]["model"] = linear_model.SGDRegressor()
# reg_models["gradient_descent"]["params"] = {
#     "loss": ["squared_loss", "huber", "epsilon_insensitive", "squared_epsilon_insensitive"],
#     "penalty": ["l2", "l1", "elasticnet"]
# }


reg_models["svm"] = {}
reg_models["svm"]["model"] = svm.SVR()
reg_models["svm"]["params"] = {
    "kernel": ["poly", "rbf"],
    "C": [0.5, 1.0, 1.2, 1.5],
    "gamma": [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
}

reg_models["knn"] = {}
reg_models["knn"]["model"] = neighbors.KNeighborsRegressor()
reg_models["knn"]["params"] = {
    "n_neighbors": [3, 5, 7, 15]
}

reg_models["random_forest"] = {}
reg_models["random_forest"]["model"] = ensemble.RandomForestRegressor()
reg_models["random_forest"]["params"] = {
    "n_estimators": [100, 200, 500]
}

# reg_models["gaussian_process"] = {}
# reg_models["gaussian_process"]["model"] = gaussian_process.GaussianProcessRegressor()
# reg_models["gaussian_process"]["params"] = {
#     "n_restarts_optimizer": [0, 1, 2]
# }
#

reg_models["decision_tree"] = {}
reg_models["decision_tree"]["model"] = tree.DecisionTreeRegressor()
reg_models["decision_tree"]["params"] = {
    "splitter": ["best", "random"],
    "min_samples_split": [2, 3, 4]
}

mean_scores = []
std_scores = []
for score in constants.CLASSIFIERS_SCORES:
    mean_scores.append(score + "_mean")
    std_scores.append(score + "_std")

if not os.path.exists("regressors"):
    os.makedirs("regressors")

divideFold = KFold(10, random_state = constants.RANDOM_STATE, shuffle = True)

regressors = {}
targets = {}
for clf in constants.CLASSIFIERS:
    for preprocess in ['None'] + constants.PRE_PROCESSES:
        regressors[clf] = {}
        regressors[clf][preprocess] = {}
        # Getting target value per classifier and preprocessor
        targets[clf] = {}
        targets[clf][preprocess] = data.query("classifier == '{}' and preprocesses == '{}'".format(clf, preprocess))
        score = CLF_SCORE
        regressors[clf][preprocess][score] = {}
        values = check_array(targets[clf][preprocess].drop(["name", "classifier", "preprocesses", *mean_scores, *std_scores], axis = 1).round(10).clip(-1e11,1e11).to_numpy().astype(np.float64))
        target = targets[clf][preprocess][score + "_mean"].astype(np.float64).to_numpy(dtype = 'float64')
        # import pdb; pdb.set_trace()
        for reg in reg_models.keys():
            opt_reg = GridSearchCV(reg_models[reg]["model"], reg_models[reg]["params"], cv = 5, error_score = 0, scoring = 'neg_mean_squared_error')
            model =  opt_reg.fit(values, target)
            # model =  opt_reg.fit(values[100:200], target[100:200])
            # model =  opt_reg.fit(values[20:25], target[20:25])
            print("Best parameters set [{}] found on development set:".format(reg))
            print(model.best_params_, "\n")
            print("Grid scores on development set:")
            means = model.cv_results_['mean_test_score']
            stds = model.cv_results_['std_test_score']
            for mean, std, params in zip(means, stds, clf.cv_results_['params']):
                print("%0.3f (+/-%0.03f) for %r"
                      % (mean, std * 2, params))
            import pdb; pdb.set_trace()
