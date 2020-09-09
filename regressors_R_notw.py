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


db = DBHelper()

SCORE_COLUMNS = ["name"] + constants.CLASSIFIERS

SCORES = ["max_error", "mean_absolute_error", "r2_score", "median_absolute_error", "mean_squared_error"]


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

metadata_means = {feature: np.mean(metadata[feature]) for feature in metadata.columns if feature != "name"}
metadata.fillna(value = metadata_means, inplace = True)

data = pd.merge(metadata, scores, on = "name")

data = data[data.preprocesses.isin(constants.PRE_PROCESSES + ["None"]) & data.classifier.isin(constants.CLASSIFIERS)]
combinations_strings = ["{}+{}".format(pp, clf) for pp in ["None"] + constants.PRE_PROCESSES
                                                for clf in constants.CLASSIFIERS]

reg_models = {}
reg_models["ann"] = lambda: R_Model(neuralnet.neuralnet)
reg_models["cart"] = lambda: R_Model(rpart.rpart)
reg_models["randomForest"] = lambda: R_Model(randomForest.randomForest)
reg_models["svm"] = lambda: SVR()
reg_models["dwnn"] = lambda: KNN()
reg_models["random"] = lambda: Random(random_seed = rand_state)
reg_models["default"] = lambda: Default()

mean_scores = []
std_scores = []
for score in constants.CLASSIFIERS_SCORES:
    mean_scores.append(score + "_mean")
    std_scores.append(score + "_std")

if not os.path.exists("regressors"):
    os.makedirs("regressors")

divideFold = KFold(10, random_state = constants.RANDOM_STATE, shuffle = True)

targets = {}
for clf in constants.CLASSIFIERS:
    for preprocess in constants.PRE_PROCESSES:
        # Getting target value per classifier and preprocessor
        combination = combinations[(combinations.classifier == clf) & (combinations.preprocesses == preprocess)]
        targets[clf] = {}
        targets[clf][preprocess] = data[(data.classifier == clf) & (data.preprocesses == preprocess)]
        for score in constants.CLASSIFIERS_SCORES:
            target = targets[clf][preprocess][score + "_mean"].values
            values = targets[clf][preprocess].drop(["name", "classifier", "preprocesses", *mean_scores, *std_scores], axis = 1)
            for reg in reg_models.keys():
                count_models = 0
                for train_indx, test_indx in divideFold.split(values):
                    model =  reg_models[reg]()
                    model.fit(values.iloc[train_indx], target[train_indx])
                    results = []; result_labels = [];
                    result_labels.append("name"); results.append(reg);
                    result_labels.append("combination_id"); results.append(int(combination.id));
                    result_labels.append("score"); results.append(score);
                    result_labels.append("model_id"); results.append(count_models);
                    for reg_score in ["mean_squared_error"]:
                        result_labels.append(reg_score)
                        result = model.predict(values.iloc[test_indx])
                        results.append(np.mean(result))
                    pickle.dump(model, open("regressors/{}_{}_{}_{}_{}.pickle".format(
                                reg, clf, preprocess, score, count_models), "wb"))
                    count_models += 1
                    db.add_regressor_preperformance_record(result_labels, results)
                print("- Finished with {} {} {} {}".format(reg, score, clf, preprocess))
