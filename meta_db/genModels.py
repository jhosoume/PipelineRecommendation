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

np.random.seed(constants.RANDOM_STATE)

SCORES = ["recall_micro", "recall_macro", "accuracy", "precision_micro",
          "precision_macro", "f1_micro", "f1_macro", "balanced_accuracy", "average_precision", "roc_auc"]

SCORES = constants.CLASSIFIERS_SCORES

datasets = [f for f in listdir(config["dataset"]["folder"])
                if ( isfile(join(config["dataset"]["folder"], f)) and
                   ( f.endswith("json") or f.endswith("arff") ) )]

db = DBHelper()
le = preprocessing.LabelEncoder()

for dataset in datasets:
    name = dataset[:-5]
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
    models = {}
    print("Generating models for", name)

    # Creating SVM model using default parameters
    svm_clf = svm.SVC(gamma = "auto").fit(values, target)
    models["svm"] = svm_clf # Actually not needed, the cv does the training again

    # Creating LogisticRegression model using default parameters
    lg_clf = linear_model.LogisticRegression(random_state = constants.RANDOM_STATE, solver = 'lbfgs').fit(values, target)
    models["logistic_regression"] = lg_clf

    # Creating Linear Discriminant model using default parameters
    lineardisc_clf = discriminant_analysis.LinearDiscriminantAnalysis().fit(values, target)
    models["linear_discriminant"] = lineardisc_clf

    # Creating KNN model using default parameters
    neigh_clf = neighbors.KNeighborsClassifier().fit(values, target)
    models["kneighbors"] = neigh_clf

    # Creating CART model using default parameters
    dectree_clf = tree.DecisionTreeClassifier(random_state = constants.RANDOM_STATE).fit(values, target)
    models["decision_tree"] = dectree_clf

    # Creating Gaussian Naive Bayes model using default parameters
    gaussian_clf = naive_bayes.GaussianNB().fit(values, target)
    models["gaussian_nb"] = gaussian_clf

    # Creating Random Forest model using default parameters
    random_forest_clf = ensemble.RandomForestClassifier(n_estimators = 100).fit(values, target)
    models["random_forest"] = random_forest_clf

    # Creating Gradient Boosting model using default parameters
    gradient_boost_clf = ensemble.GradientBoostingClassifier().fit(values, target)
    models["gradient_boosting"] = gradient_boost_clf

    # Creating Neual Network model using default parameters
    neural_network_clf = neural_network.MLPClassifier().fit(values, target)
    models["neural_network"] = neural_network_clf

    # Calculate mean and std for CV = 10
    for model in models.keys():
        if (name, model) in db.get_models_indx():
            continue
        print("[{}] Calculating scores for model {}".format(name, model))
        cv_results = cross_validate(models[model], values, target, cv = 10, scoring = SCORES)
        results = []; result_labels = [];
        result_labels.append("name"); results.append(name);
        result_labels.append("model"); results.append(model);
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
        db.add_model_record(result_labels, results)
        print("- Finished with {}".format(name))
