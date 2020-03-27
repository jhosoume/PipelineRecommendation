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

import constants
from Default import Default
from Random import Random
from meta_db.db.DBHelper import DBHelper

db = DBHelper()

SCORE_COLUMNS = ["name"] + constants.CLASSIFIERS

SCORES = ["max_error", "mean_absolute_error", "r2_score", "median_absolute_error"]


data = pd.DataFrame(db.get_all_metadata(), columns = db.metadata_columns()).drop("id", axis = 1)
models = pd.DataFrame(db.get_all_models(), columns = db.models_columns()).drop("id", axis = 1)
preperformance = pd.DataFrame(db.get_all_preperformance(), columns = db.preperformance_columns()).drop("id", axis = 1)
# Not null preperformance
preperformance = preperformance[~preperformance.isnull().any(axis = 1)]

combinations = pd.DataFrame(db.get_all_combinations(), columns = db.combinations_columns())

data_means = {feature: np.mean(data[feature]) for feature in data.columns if feature != "name"}
data.fillna(value = data_means, inplace = True)

reg_models = {}

# reg_models["neural_network"] = {}
# reg_models["neural_network"]["model"] = neural_network.MLPRegressor()
# reg_models["neural_network"]["params"] = {
#     "hidden_layer_sizes": [50, 100, 150, 200],
#     "activation": ["logistic", "tanh", "relu"],
#     "solver": ["lbfgs", "sgd", "adam"]
# }


reg_models["ridge"] = {}
reg_models["ridge"]["model"] = linear_model.Ridge()
reg_models["ridge"]["params"] = {
    "alpha": [0.5, 1.0, 1.5],
    "solver": ["svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga"]
}

reg_models["gradient_descent"] = {}
reg_models["gradient_descent"]["model"] = linear_model.SGDRegressor()
reg_models["gradient_descent"]["params"] = {
    "loss": ["squared_loss", "huber", "epsilon_insensitive", "squared_epsilon_insensitive"],
    "penalty": ["l2", "l1", "elasticnet"]
}


reg_models["svm"] = {}
reg_models["svm"]["model"] = svm.SVR(gamma = "auto")
reg_models["svm"]["params"] = {
    "kernel": ["linear", "poly", "rbf", "sigmoid", "precomputed"],
    "C": [0.5, 1.0, 1.2, 1.5]
}

reg_models["knn"] = {}
reg_models["knn"]["model"] = neighbors.KNeighborsRegressor()
reg_models["knn"]["params"] = {
    "n_neighbors": [5, 8, 10, 15],
    "algorithm": ["ball_tree", "kd_tree", "brute"]
}

reg_models["random_forest"] = {}
reg_models["random_forest"]["model"] = ensemble.RandomForestRegressor()
reg_models["random_forest"]["params"] = {
    "n_estimators": [5, 10, 12, 15]
}

reg_models["gaussian_process"] = {}
reg_models["gaussian_process"]["model"] = gaussian_process.GaussianProcessRegressor()
reg_models["gaussian_process"]["params"] = {
    "n_restarts_optimizer": [0, 1, 2]
}


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
    for preprocess in constants.PRE_PROCESSES:
        regressors[clf] = {}
        regressors[clf][preprocess] = {}
        # Getting target value per classifier and preprocessor
        combination = combinations[(combinations.classifier == clf) & (combinations.preprocesses == preprocess)]
        targets[clf] = {}
        targets[clf][preprocess] = preperformance[preperformance.combination_id == int(combination.id)]
        targets[clf][preprocess] = pd.merge(data, targets[clf][preprocess], on = "name")
        for score in constants.CLASSIFIERS_SCORES:
            regressors[clf][preprocess][score] = {}
            values = targets[clf][preprocess].drop(["name", *mean_scores, *std_scores], axis = 1).values
            target = targets[clf][preprocess][score + "_mean"].values
            for reg in reg_models.keys():
                count_models = 0
                # model =  reg_models[reg].fit(values, target)
                # regressors[clf][score][reg] = model
                for train_indx, test_indx in divideFold.split(values):
                    opt_reg = GridSearchCV(reg_models[reg]["model"], reg_models[reg]["params"], cv = 5)
                    model =  opt_reg.fit(values[train_indx], target[train_indx])
                    results = []; result_labels = [];
                    result_labels.append("name"); results.append(reg + "optimized");
                    result_labels.append("combination_id"); results.append(int(combination.id));
                    result_labels.append("score"); results.append(score);
                    result_labels.append("model_id"); results.append(count_models);
                    for reg_score in SCORES:
                        result_labels.append(reg_score)
                        result = getattr(metrics, reg_score)(target[test_indx], model.predict(values[test_indx]))
                        #if result > 100000 and reg_score == "mean_absolute_error":
                        #    print("REPORTING")
                        # import pdb; pdb.set_trace()
                        results.append(result)
                    pickle.dump(model, open("regressors/{}_{}_{}_{}_{}.pickle".format(
                                reg, clf, preprocess, score, count_models), "wb"))
                    count_models += 1
                    db.add_regressor_preperformance_record(result_labels, results)
                print("- Finished with {} {} {}".format(reg, score, clf))
