import sys
import mysql
from os import listdir
from os.path import isfile, join
import pandas as pd
import numpy as np

from sklearn import svm, linear_model, discriminant_analysis, neighbors
from sklearn import tree, naive_bayes, ensemble, neural_network
from sklearn.model_selection import KFold
from sklearn import metrics
from sklearn.metrics import get_scorer
from sklearn import preprocessing
from scipy.io import arff as arff_io

from imblearn.over_sampling import ADASYN

import constants
import logging
import time
from config import config

from meta_db.db.DBHelper import DBHelper

SCORES = ["recall_micro", "recall_macro", "accuracy", "precision_micro",
          "precision_macro", "f1_micro", "f1_macro", "fit_time", "score_time"]

SCORERS = {
    "recall_micro":    lambda y_true, y_pred: metrics.recall_score(y_true, y_pred, average = "micro"),
    "recall_macro":    lambda y_true, y_pred: metrics.recall_score(y_true, y_pred, average = "macro"),
    "accuracy":        lambda y_true, y_pred: metrics.accuracy_score(y_true, y_pred),
    "precision_micro": lambda y_true, y_pred: metrics.precision_score(y_true, y_pred, average = "micro"),
    "precision_macro": lambda y_true, y_pred: metrics.precision_score(y_true, y_pred, average = "macro"),
    "f1_micro":        lambda y_true, y_pred: metrics.f1_score(y_true, y_pred, average = "micro"),
    "f1_macro":        lambda y_true, y_pred: metrics.f1_score(y_true, y_pred, average = "macro")
}

MODELS = {}

LOG_FILENAME = "preprocesses.log"

logging.basicConfig(filename = LOG_FILENAME, level = logging.DEBUG)

# Creating SVM model using default parameters
svm_clf = svm.SVC(gamma = "auto")
MODELS["svm"] = svm_clf # Actually not needed, the cv does the training again

# Creating LogisticRegression model using default parameters
lg_clf = linear_model.LogisticRegression(random_state = constants.RANDOM_STATE, solver = 'lbfgs', n_jobs = -1)
MODELS["logistic_regression"] = lg_clf

# Creating Linear Discriminant model using default parameters
lineardisc_clf = discriminant_analysis.LinearDiscriminantAnalysis()
MODELS["linear_discriminant"] = lineardisc_clf

# Creating KNN model using default parameters
neigh_clf = neighbors.KNeighborsClassifier(n_jobs = -1)
MODELS["kneighbors"] = neigh_clf

# Creating CART model using default parameters
dectree_clf = tree.DecisionTreeClassifier(random_state = constants.RANDOM_STATE)
MODELS["decision_tree"] = dectree_clf

# Creating Gaussian Naive Bayes model using default parameters
gaussian_clf = naive_bayes.GaussianNB()
MODELS["gaussian_nb"] = gaussian_clf

# Creating Random Forest model using default parameters
random_forest_clf = ensemble.RandomForestClassifier(n_estimators = 100, n_jobs = -1)
MODELS["random_forest"] = random_forest_clf

# Creating Gradient Boosting model using default parameters
gradient_boost_clf = ensemble.GradientBoostingClassifier()
MODELS["gradient_boosting"] = gradient_boost_clf

# Creating Neual Network model using default parameters
neural_network_clf = neural_network.MLPClassifier()
MODELS["neural_network"] = neural_network_clf


np.random.seed(constants.RANDOM_STATE)

PRE_PROCESSOR = ADASYN(random_state = constants.RANDOM_STATE)
PRE_PROCESSOR_NAME = "ADASYN"

datasets = [f for f in listdir(config["dataset"]["folder"])
                if ( isfile(join(config["dataset"]["folder"], f)) and
                   ( f.endswith("json") or f.endswith("arff") ) )]

divideFold = KFold(10, random_state = constants.RANDOM_STATE, shuffle = True)

db = DBHelper()
le = preprocessing.LabelEncoder()
for dataset in datasets:
    dataset = "48_tae.arff"
    name = dataset[:-5]
    print("[{}]".format(name))
    if dataset.endswith("json"):
        data = pd.read_json(config["dataset"]["folder"] + dataset)
    elif dataset.endswith("arff"):
        data = arff_io.loadarff(config["dataset"]["folder"] + dataset)
        data = pd.DataFrame(data[0])
    target = data["class"].values
    if target.dtype == np.object:
        le.fit(target)
        target = le.transform(target)
    values = data.drop("class", axis = 1)
    # Check if any is a string, some classifiers only deals with numeric data
    for dtype, key in zip(values.dtypes, values.keys()):
        if dtype == np.object:
            le.fit(values[key].values)
            values[key] = le.transform(values[key].values)
    values = values.values

    cv_results = {}
    for model in MODELS.keys():
        cv_results[model] = {}
        for score in SCORES:
            cv_results[model][score] = []
    iter = 0
    for train_indx, test_indx in divideFold.split(values):
        try:
            new_values, new_target = PRE_PROCESSOR.fit_resample(values[train_indx], target[train_indx])
        except (RuntimeError, ValueError) as err:
            print("\t\tCould not perform preprocessing. {}".format(err))
            logging.info("Could not perform preprocessing. [{}, {}, {}]".format(name, model, PRE_PROCESSOR_NAME))
            logging.error(err)
            continue

        for model in MODELS.keys():
            print("\t[{}] Calculating scores for model {} iter {}".format(name, model, iter))
            combination_id = db.get_combination(model, PRE_PROCESSOR_NAME)[0]
            print("\t\tUsing preprocessor {}.".format(PRE_PROCESSOR_NAME))
            if (name, combination_id) in db.get_preperformance_done():
                print("\tAlready done! Skiping...")
                continue

            try:
                start_fit = time.time()
                clf = MODELS[model].fit(new_values, new_target)
                stop_fit = time.time()
                cv_results[model]["fit_time"].append(stop_fit - start_fit)
                target_pred = target[test_indx]
                start_score = time.time()
                target_pred = clf.predict(values[test_indx])
                stop_score = time.time()
                cv_results[model]["score_time"].append(stop_score - start_score)
            except Exception as err:
                logging.debug("Could not execute fit of dataset {} with classifier {} and {}.". format(name, model, PRE_PROCESSOR_NAME))
                logging.error(err)
                continue
            for score in SCORES[:-2]:
                scorer = SCORERS[score]
                res = scorer(target[test_indx], target_pred)
                cv_results[model][score].append(res)
        iter += 1

    for model in MODELS.keys():
        print("\t[{}] FINISHING scores for model {}".format(name, model))
        results = []; result_labels = [];
        result_labels.append("name"); results.append(name);
        combination_id = db.get_combination(model, PRE_PROCESSOR_NAME)[0]
        result_labels.append("combination_id"); results.append(combination_id);
        for rlabel in cv_results[model].keys():
            if rlabel.startswith("test"):
                rlabel_db = rlabel[5:]
            else:
                rlabel_db = rlabel
            result_labels.append(rlabel_db + "_mean")
            results.append(np.mean(cv_results[model][rlabel]))
            result_labels.append(rlabel_db + "_std")
            results.append(np.std(cv_results[model][rlabel]))
        partial_res = np.array(results[1:]); result_labels = np.array(result_labels)
        if np.any(np.isnan(partial_res)):
            not_nan = np.insert(~np.isnan(partial_res), 0, True, axis = 0)
            results = np.array(results)[not_nan].tolist()
            result_labels = result_labels[not_nan].tolist()
        db.add_preperformance_record(result_labels, results)
