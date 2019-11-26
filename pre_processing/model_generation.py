import sys
import mysql
from os import listdir
from os.path import isfile, join
import pandas as pd
import numpy as np

from sklearn import svm, linear_model, discriminant_analysis, neighbors
from sklearn import tree, naive_bayes, ensemble, neural_network
from sklearn.model_selection import cross_validate
from sklearn.metrics import SCORERS
from sklearn import preprocessing
from scipy.io import arff as arff_io

import constants
from config import config

from meta_db.db.DBHelper import DBHelper

SCORES = ["recall_micro", "recall_macro", "accuracy", "precision_micro",
          "precision_macro", "f1_micro", "f1_macro"]

MODELS = {}

# Creating SVM model using default parameters
svm_clf = svm.SVC(gamma = "auto")
MODELS["svm"] = svm_clf # Actually not needed, the cv does the training again

# Creating LogisticRegression model using default parameters
lg_clf = linear_model.LogisticRegression(random_state = constants.RANDOM_STATE, solver = 'lbfgs')
MODELS["logistic_regression"] = lg_clf

# Creating Linear Discriminant model using default parameters
lineardisc_clf = discriminant_analysis.LinearDiscriminantAnalysis()
MODELS["linear_discriminant"] = lineardisc_clf

# Creating KNN model using default parameters
neigh_clf = neighbors.KNeighborsClassifier()
MODELS["kneighbors"] = neigh_clf

# Creating CART model using default parameters
dectree_clf = tree.DecisionTreeClassifier(random_state = constants.RANDOM_STATE)
MODELS["decision_tree"] = dectree_clf

# Creating Gaussian Naive Bayes model using default parameters
gaussian_clf = naive_bayes.GaussianNB()
MODELS["gaussian_nb"] = gaussian_clf

# Creating Random Forest model using default parameters
random_forest_clf = ensemble.RandomForestClassifier(n_estimators = 100)
MODELS["random_forest"] = random_forest_clf

# Creating Gradient Boosting model using default parameters
gradient_boost_clf = ensemble.GradientBoostingClassifier()
MODELS["gradient_boosting"] = gradient_boost_clf

# Creating Neual Network model using default parameters
neural_network_clf = neural_network.MLPClassifier()
MODELS["neural_network"] = neural_network_clf

def calculate(name, train_values, train_target, test_values, test_target, preprocess, models = MODELS):
    db = DBHelper()
    # Calculate mean and std for CV = 10
    for model in models.keys():
        print("\t[{}] Calculating scores for model {}".format(name, model))
        combination_id = db.get_combination(model, preprocess)[0]
        if (name, combination_id) in db.get_preperformance_done():
            print("\tAlready done! Skiping...")
            continue
        models[model] = models[model].fit(train_values, train_target)
        cv_results = cross_validate(models[model], test_values, test_target, cv = 10, scoring = SCORES)
        results = []; result_labels = [];
        result_labels.append("name"); results.append(name);
        result_labels.append("combination_id"); results.append(combination_id);
        for rlabel in cv_results.keys():
            if rlabel.startswith("test"):
                rlabel_db = rlabel[5:]
            else:
                rlabel_db = rlabel
            # print("[{}, {}] Scores {}".format(name, model, rlabel_db))
            result_labels.append(rlabel_db + "_mean")
            results.append(np.mean(cv_results[rlabel]))
            result_labels.append(rlabel_db + "_std")
            results.append(np.std(cv_results[rlabel]))
        db.add_preperformance_record(result_labels, results)
        print("- Finished with {}".format(name))
