import sys
import os
import pandas as pd
import numpy as np
import pickle

from sklearn import svm, linear_model, discriminant_analysis, neighbors
from sklearn import tree, naive_bayes, ensemble, neural_network, gaussian_process
from sklearn.model_selection import cross_validate, KFold
from sklearn import metrics
from sklearn import preprocessing

import constants
from Default import Default
from Random import Random
from meta_db.db.DBHelper import DBHelper

db = DBHelper()

SCORE_COLUMNS = ["name"] + constants.CLASSIFIERS

SCORES = ["max_error", "mean_absolute_error", "r2_score", "median_absolute_error", "mean_squared_error"]


data = pd.DataFrame(db.get_all_metadata(), columns = db.metadata_columns()).drop("id", axis = 1)
models = pd.DataFrame(db.get_all_models(), columns = db.models_columns()).drop("id", axis = 1)

data_means = {feature: np.mean(data[feature]) for feature in data.columns if feature != "name"}
data.fillna(value = data_means, inplace = True)

reg_models = {}
reg_models["gradient_boosting"] = ensemble.GradientBoostingRegressor()
reg_models["neural_network"] = neural_network.MLPRegressor()
reg_models["ridge"] = linear_model.Ridge()
reg_models["gradient_descent"] = linear_model.SGDRegressor()
reg_models["svm"] = svm.SVR(gamma = "auto")
reg_models["knn"] = neighbors.KNeighborsRegressor()
reg_models["random_forest"] = ensemble.RandomForestRegressor()
reg_models["gaussian_process"] = gaussian_process.GaussianProcessRegressor()
reg_models["decision_tree"] = tree.DecisionTreeRegressor()
reg_models["random"] = Random()
reg_models["default"] = Default()

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
    regressors[clf] = {}
    targets[clf] = pd.DataFrame(db.get_model_record_per_model(clf), columns = db.models_columns())
    targets[clf] = pd.merge(data, targets[clf], on = "name")
    for score in constants.CLASSIFIERS_SCORES:
        regressors[clf][score] = {}
        values = targets[clf].drop(["name", "model", *mean_scores, *std_scores, "id"], axis = 1).values
        target = targets[clf][score + "_mean"].values
        for reg in reg_models.keys():
            count_models = 0
            # model =  reg_models[reg].fit(values, target)
            # regressors[clf][score][reg] = model
            for train_indx, test_indx in divideFold.split(values):
                model =  reg_models[reg].fit(values[train_indx], target[train_indx])
                results = []; result_labels = [];
                result_labels.append("name"); results.append(reg);
                result_labels.append("classifier"); results.append(clf);
                result_labels.append("score"); results.append(score);
                result_labels.append("model_id"); results.append(count_models);
                for reg_score in SCORES:
                    result_labels.append(reg_score)
                    result = getattr(metrics, reg_score)(target[test_indx], model.predict(values[test_indx]))
                    #if result > 100000 and reg_score == "mean_absolute_error":
                    #    print("REPORTING")
                    # import pdb; pdb.set_trace()
                    results.append(result)
                pickle.dump(model, open("regressors/{}_{}_{}_{}.pickle".format(
                            reg, clf, score, count_models), "wb"))
                count_models += 1
                db.add_regressor_record(result_labels, results)
            print("- Finished with {} {} {}".format(reg, score, clf))
