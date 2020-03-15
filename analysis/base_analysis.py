import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm, linear_model, discriminant_analysis, neighbors
from sklearn import tree, naive_bayes, ensemble, neural_network, gaussian_process
from sklearn.model_selection import cross_validate, KFold
from sklearn import metrics
from sklearn import preprocessing

import constants
from Default import Default
from Random import Random
from meta_db.db.DBHelper import DBHelper

SCORE = "accuracy_mean"

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
scores = scores[scores.classifier != "neural_network"]
models = models[models.classifier != "neural_network"]

metadata_means = {feature: np.mean(metadata[feature]) for feature in metadata.columns if feature != "name"}
metadata.fillna(value = metadata_means, inplace = True)

data = pd.merge(metadata, scores, on = "name")

if not os.path.exists("analysis/plots"):
    os.makedirs("analysis/plots")
if not os.path.exists("analysis/plots/base_analysis"):
    os.makedirs("analysis/plots/base_analysis")

mean_scores = []
std_scores = []
for score in constants.CLASSIFIERS_SCORES:
    mean_scores.append(score + "_mean")
    std_scores.append(score + "_std")


reg_models = {}
reg_models["neural_network"] = lambda: neural_network.MLPRegressor()
reg_models["ridge"] = lambda: linear_model.Ridge()
reg_models["gradient_descent"] = lambda: linear_model.SGDRegressor()
reg_models["svm"] = lambda: svm.SVR(gamma = "auto")
reg_models["knn"] = lambda: neighbors.KNeighborsRegressor()
reg_models["random_forest"] = lambda: ensemble.RandomForestRegressor()
reg_models["gaussian_process"] = lambda: gaussian_process.GaussianProcessRegressor()
reg_models["decision_tree"] = lambda: tree.DecisionTreeRegressor()
reg_models["random"] = lambda: Random()
reg_models["default"] = lambda: Default()

results = {baseline: [{reg: 0 for reg in reg_models.keys()} for num in range(10)]
            for baseline in ["random", "default"]}

divideFold = KFold(10, random_state = constants.RANDOM_STATE, shuffle = True)

def filter_dataset(database):
    datasets_filtered = []
    total_combinations = len(constants.CLASSIFIERS) * len(constants.PRE_PROCESSES + ['None'])
    for dataset in database.name.unique():
        split = database[database.name == dataset]
        if len(split) == total_combinations:
            datasets_filtered.append(dataset)
    return datasets_filtered

datasets = pd.Series(filter_dataset(data))

for baseline in ["random", "default"]:
    for regressor_type in list(reg_models.keys())[:-2]:
        for train_indx, test_indx in divideFold.split(datasets):
            targets = data[data.name.isin(list(datasets.iloc[train_indx]))]
            models = {}
            baseline_models = {}
            for clf in constants.CLASSIFIERS:
                for preprocess in (constants.PRE_PROCESSES + ['None']):
                    models["{}+{}".format(preprocess, clf)] = reg_models[regressor_type]()
                    baseline_models["{}+{}".format(preprocess, clf)] = reg_models[baseline]()
                    target = targets.query("classifier == '{}' and preprocesses == '{}'".format(clf, preprocess))
                    meta_target = target.drop(["name", "classifier", "preprocesses", *mean_scores, *std_scores], axis = 1).values
                    label_target = target[SCORE].values
                    models["{}+{}".format(preprocess, clf)].fit(meta_target, label_target)
                    baseline_models["{}+{}".format(preprocess, clf)].fit(meta_target, label_target)
            tests = data[data.name.isin(list(datasets.iloc[test_indx]))]
            reg_results = {}
            for reg in models:
                ref_results[reg] = models[reg].predict()
















def histogram(score = "accuracy_mean"):
    pp_clf_count = {"{}+{}".format(comb.preprocesses, comb.classifier):0 for indx, comb in combinations.iterrows()}
    for clf in models.classifier.unique():
        pp_clf_count["None+{}".format(clf)] = 0

    for dataset in models.name.unique():
        result_dataset = data.query("name == '{}'".format(dataset))
        max_result = result_dataset[result_dataset[score] == result_dataset[score].max()]
        for indx, result in max_result.iterrows():
            pp_clf_count["{}+{}".format(result.preprocesses, result.classifier)] += 1

    # fig = plt.figure(figsize = (12, 4))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.bar(pp_clf_count.keys(), pp_clf_count.values())
    plt.xlabel("PreProcesse + Classifier", fontsize = 12, fontweight = 'bold')
    plt.ylabel(score.replace("_", " ").capitalize(), fontsize = 12, fontweight = 'bold')
    plt.xticks(rotation=90)
    plt.grid(True, alpha = 0.5, linestyle = 'dotted')
    plt.gcf().subplots_adjust(bottom=0.60)

for score in constants.CLASSIFIERS_SCORES:
    histogram(score = score + "_mean")
    plt.savefig("analysis/plots/winnings/" + score + ".png", dpi = 100)
