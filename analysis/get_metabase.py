import os
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
import json

from sklearn import svm, linear_model, discriminant_analysis, neighbors
from sklearn import tree, naive_bayes, ensemble, neural_network, gaussian_process
from sklearn.model_selection import cross_validate, KFold
from sklearn import metrics
from sklearn import preprocessing

import constants
from Default import Default
from Random import Random
from meta_db.db.DBHelper import DBHelper

import plotly.io as pio
import plotly.express as px
import plotly.graph_objects as go


pio.templates.default = "plotly_white"

SCORE = "balanced_accuracy_mean"

grey_palette = ['rgb(208, 209, 211)',
                'rgb(185, 191, 193)',
                'rgb(137, 149, 147)',
                'rgb(44, 54, 60)',
                'rgb(3, 3, 3)'
               ]

translator = {
    "svm": "SVM",
    "logistic_regression": "LG",
    "linear_discriminant": "LD",
    "kneighbors": "kNN",
    "decision_tree": "DT",
    "gaussian_nb": "GNB",
    "random_forest": "RF",
    "gradient_boosting": "GB",
    "neural_network": "NN",
    "knn": "kNN",
    "Svm": "SVM",
    "random": "Random",
    "default": "Default"
}

db = DBHelper()

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
scores = scores[scores.classifier != "neural_network"]
scores = scores.drop(columns = ["recall_micro_mean", "recall_micro_std", "precision_macro_mean", "precision_macro_std", "recall_macro_std", "recall_macro_mean", "accuracy_mean", "accuracy_std","precision_micro_mean", "precision_micro_std", "f1_micro_mean", "f1_micro_std", "f1_macro_std", "f1_macro_mean","fit_time_mean", "fit_time_std", "score_time_mean", "score_time_std" ])
models = models[models.classifier != "neural_network"]

metadata_means = {feature: np.mean(metadata[feature]) for feature in metadata.columns if feature != "name"}
metadata.fillna(value = metadata_means, inplace = True)

data = pd.merge(metadata, scores, on = "name")

data = data[data.preprocesses.isin(constants.PRE_PROCESSES + ["None"]) & data.classifier.isin(constants.CLASSIFIERS)]

wins = {"{}+{}".format(pproc, clf):0 for pproc in ["None"] + constants.PRE_PROCESSES for clf in constants.CLASSIFIERS}
for dataset in models.name.unique():
    result_dataset = data.query("name == '{}'".format(dataset))
    max_result = result_dataset[result_dataset[SCORE] == result_dataset[SCORE].max()]
    # Note that results can be similar, so a dataset is included multiple times
    for indx, result in max_result.iterrows():
        wins["{}+{}".format(result.preprocesses, result.classifier)] += 1
default_max_baseline = max(wins, key = lambda key: wins[key])
print("Default is:", default_max_baseline)

if not os.path.exists("analysis/plots"):
    os.makedirs("analysis/plots")
if not os.path.exists("analysis/plots/base_analysis"):
    os.makedirs("analysis/plots/base_analysis")

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
                if len(split[split.classifier == clf][split.preprocesses == pp]) < 1:
                    keep = False
        if keep:
            datasets_filtered.append(dataset)
    return datasets_filtered

datasets = pd.Series(filter_dataset(data))
print("Num datasets:", len(datasets))

# metabase_columns = metadata.columns + ["{}+{}".format(pp, clf)
#                                             for pp in ["None"] + constants.PRE_PROCESSES
#                                             for clf in constants.CLASSIFIERS]

list_datasets = []
for dataset in datasets:
    meta_features = metadata[metadata.name == dataset]
    dataset_dict = {col:value for col, value in zip(list(meta_features), list(meta_features.values[0]))}
    for pp in ['None'] + constants.PRE_PROCESSES:
        for clf in constants.CLASSIFIERS:
            dataset_dict["{}+{}".format(pp, clf)] = float(scores[(scores.name == dataset) & (scores.preprocesses == pp) & (scores.classifier == clf)]["balanced_accuracy_mean"])
    list_datasets.append(dataset_dict)


df = pd.DataFrame(list_datasets)
df.to_csv("metabase.csv", sep = ",")
