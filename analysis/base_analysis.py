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

np.random.seed(constants.RANDOM_STATE)

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

wins = {"{}+{}".format(pproc, clf):0 for pproc in ["None"] + constants.PRE_PROCESSES for clf in constants.CLASSIFIERS}
for dataset in models.name.unique():
    result_dataset = data.query("name == '{}'".format(dataset))
    max_result = result_dataset[result_dataset[SCORE] == result_dataset[SCORE].max()]
    # Note that results can be similar, so a dataset is included multiple times
    for indx, result in max_result.iterrows():
        wins["{}+{}".format(result.preprocesses, result.classifier)] += 1
default_max_baseline = max(wins, key = lambda key: wins[key])

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
reg_models["random_forest"] = lambda: ensemble.RandomForestRegressor(random_state = constants.RANDOM_STATE)
reg_models["gaussian_process"] = lambda: gaussian_process.GaussianProcessRegressor()
reg_models["decision_tree"] = lambda: tree.DecisionTreeRegressor(random_state = constants.RANDOM_STATE)
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
results = {}



for baseline in ["default", "random"]:
    results[baseline] = {}

for regressor_type in constants.REGRESSORS[:-2]:
    # results[baseline][regressor_type] = {}
    kfold = 0
    for baseline in ["default", "random"]:
        results[baseline][regressor_type] = []
    for train_indx, test_indx in divideFold.split(datasets):
        # results[baseline][regressor_type][kfold] = []
        targets = data[data.name.isin(list(datasets.iloc[train_indx]))]
        models = {}
        baseline_models = {}
        for clf in constants.CLASSIFIERS:
            for preprocess in (constants.PRE_PROCESSES + ['None']):
                models["{}+{}".format(preprocess, clf)] = reg_models[regressor_type]()
                for baseline in ["default", "random"]:
                    baseline_models["{}+{}".format(preprocess, clf)] = reg_models[baseline]()
                target = targets.query("classifier == '{}' and preprocesses == '{}'".format(clf, preprocess))
                meta_target = target.drop(["name", "classifier", "preprocesses", *mean_scores, *std_scores], axis = 1).values
                label_target = target[SCORE].values
                models["{}+{}".format(preprocess, clf)].fit(meta_target, label_target)
                baseline_models["{}+{}".format(preprocess, clf)].fit(meta_target, label_target)
        tests = data[data.name.isin(list(datasets.iloc[test_indx]))]
        for test_dataset in tests.name.unique():
            dataset_info = tests.query(
                "name == '{}'".format(test_dataset)
            )
            meta_data = dataset_info.iloc[0].drop(
                    ["name", "classifier", "preprocesses", *mean_scores, *std_scores]
                ).values.reshape(1, -1)
            true_max = dataset_info[dataset_info[SCORE] == dataset_info[SCORE].max()]
            reg_results = {}
            baseline_results = {}
            for model in models:
                reg_results[model] = models[model].predict(meta_data)
                baseline_results[model] = baseline_models[model].predict(meta_data)
            max_predicted = max(reg_results.keys(), key=(lambda key: reg_results[key]))
            pp_pred, clf_pred = max_predicted.split("+")
            max_baseline = max(baseline_results.keys(), key=(lambda key: baseline_results[key]))
            if (baseline == 'default'):
                max_baseline = default_max_baseline
            pp_base, clf_base = max_baseline.split("+")
            score_pred = dataset_info[(dataset_info.preprocesses == pp_pred) & (dataset_info.classifier == clf_pred)][SCORE]
            score_baseline = dataset_info[(dataset_info.preprocesses == pp_base) & (dataset_info.classifier == clf_base)][SCORE]
            # results[baseline][regressor_type][kfold].append(float(score_pred) - float(score_baseline))
            results[baseline][regressor_type].append(float(score_pred) - float(score_baseline))
            # if results[baseline][regressor_type][kfold][-1] == 0:
            #     import pdb; pdb.set_trace()
        kfold += 1

results[baseline]["true_max"] = []
for train_indx, test_indx in divideFold.split(datasets):
    baseline_models = {}
    for clf in constants.CLASSIFIERS:
        for preprocess in (constants.PRE_PROCESSES + ['None']):
            baseline_models["{}+{}".format(preprocess, clf)] = reg_models[baseline]()
            target = targets.query("classifier == '{}' and preprocesses == '{}'".format(clf, preprocess))
            meta_target = target.drop(["name", "classifier", "preprocesses", *mean_scores, *std_scores], axis = 1).values
            label_target = target[SCORE].values
            baseline_models["{}+{}".format(preprocess, clf)].fit(meta_target, label_target)

    tests = data[data.name.isin(list(datasets.iloc[test_indx]))]
    for test_dataset in tests.name.unique():
        dataset_info = tests.query(
            "name == '{}'".format(test_dataset)
        )
        meta_data = dataset_info.iloc[0].drop(
                ["name", "classifier", "preprocesses", *mean_scores, *std_scores]
            ).values.reshape(1, -1)
        true_max = dataset_info[dataset_info[SCORE] == dataset_info[SCORE].max()].iloc[0]["accuracy_mean"]
        baseline_results = {}
        for model in models:
            baseline_results[model] = baseline_models[model].predict(meta_data)
        max_baseline = max(baseline_results.keys(), key=(lambda key: baseline_results[key]))
        pp_base, clf_base = max_baseline.split("+")
        score_baseline = dataset_info[(dataset_info.preprocesses == pp_base) & (dataset_info.classifier == clf_base)][SCORE]
        results[baseline]["true_max"].append(float(true_max) - float(score_baseline))

def histogram(baseline = 'default'):
    # fig = plt.figure(figsize = (12, 4))
    fig = plt.figure()
    fig.suptitle(baseline, fontsize = 12, fontweight = 'bold')
    ax = fig.add_subplot(111)
    ax.bar(results[baseline].keys(), [np.sum(values) for values in results[baseline].values()])
    plt.xlabel("Regressor", fontsize = 12, fontweight = 'bold')
    plt.ylabel("Gain", fontsize = 12, fontweight = 'bold')
    plt.xticks(rotation=90)
    plt.ylim([-4, 5])
    plt.grid(True, alpha = 0.5, linestyle = 'dotted')
    plt.gcf().subplots_adjust(bottom=0.60)

for baseline in results.keys():
    histogram(baseline)
    plt.savefig("analysis/plots/base_analysis/" + baseline + ".png", dpi = 100)
