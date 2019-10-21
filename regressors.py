import sys
import os
import pandas as pd
import numpy as np
import pickle

from sklearn import svm, linear_model, discriminant_analysis, neighbors
from sklearn import tree, naive_bayes, ensemble, neural_network, gaussian_process
from sklearn.model_selection import cross_validate
from sklearn.metrics import SCORERS
from sklearn import preprocessing
from sklean import LeavePOut

import constants
from meta_db.db.DBHelper import DBHelper

db = DBHelper()

SCORE_COLUMNS = ["name"] + constants.CLASSIFIERS

SCORES = ["max_error", "neg_mean_absolute_error", "r2", "neg_median_absolute_error"]


data = pd.DataFrame(db.get_all_metadata(), columns = db.metadata_columns()).drop("id", axis = 1)
models = pd.DataFrame(db.get_all_models(), columns = db.models_columns()).drop("id", axis = 1)

data_means = {feature: np.mean(data[feature]) for feature in data.columns if feature != "name"}
data.fillna(value = data_means, inplace = True)

reg_models = {}
reg_models["neural_network"] = neural_network.MLPRegressor()
reg_models["ridge"] = linear_model.Ridge()
reg_models["gradient_descent"] = linear_model.SGDRegressor()
reg_models["svm"] = svm.SVR(gamma = "auto")
reg_models["knn"] = neighbors.KNeighborsRegressor()
reg_models["random_forest"] = ensemble.RandomForestRegressor()
reg_models["gaussian_process"] = gaussian_process.GaussianProcessRegressor()
reg_models["decision_tree"] = tree.DecisionTreeRegressor()

mean_scores = []
std_scores = []
for score in constants.CLASSIFIERS_SCORES:
    mean_scores.append(score + "_mean")
    std_scores.append(score + "_std")

if not os.path.exists("regressors"):
    os.makedirs("regressors")

regressors = {}
targets = {}
for clf in constants.CLASSIFIERS:
    regressors[clf] = {}
    targets[clf] = pd.DataFrame(db.get_model_record_per_model(clf), columns = db.models_columns())
    targets[clf] = pd.merge(data, targets[clf], on = "name")
    for score in constants.CLASSIFIERS_SCORES:
        regressors[clf][score] = {}
        values = targets[clf].drop(["name", "model", *mean_scores, *std_scores, "id"], axis = 1).values
        target = targets[clf][score + "_mean"].values
        for reg in reg_models.keys():
            model =  reg_models[reg].fit(values, target)
            regressors[clf][score][reg] = model
            # pickle.dump(model, open("regressors/{}_{}_{}.pickle".format(reg, clf, score), "wb"))
            # cv_results = cross_validate(model, values, target, cv = 10,
            #                             scoring = SCORES)
            # results = []; result_labels = [];
            # result_labels.append("name"); results.append(reg);
            # result_labels.append("classifier"); results.append(clf);
            # result_labels.append("score"); results.append(score);
            # for rlabel in cv_results.keys():
            #     if rlabel.startswith("test"):
            #         rlabel_db = rlabel[5:]
            #     else:
            #         rlabel_db = rlabel
            #     # print("[{}, {}] Scores {}".format(name, model, rlabel_db))
            #     result_labels.append(rlabel_db + "_mean")
            #     results.append(np.mean(cv_results[rlabel]))
            #     result_labels.append(rlabel_db + "_std")
            #     results.append(np.std(cv_results[rlabel]))
            # db.add_regressor_record(result_labels, results)
            # print("- Finished with {} {} {}".format(reg, score, clf))
