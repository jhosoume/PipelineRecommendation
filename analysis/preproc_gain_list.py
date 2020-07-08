import os
import pandas as pd
import numpy as np
import json
# import matplotlib.pyplot as plt

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
scores = scores[scores.classifier != "neural_network"]
models = models[models.classifier != "neural_network"]

metadata_means = {feature: np.mean(metadata[feature]) for feature in metadata.columns if feature != "name"}
metadata.fillna(value = metadata_means, inplace = True)

data = pd.merge(metadata, scores, on = "name")

data = data[data.preprocesses.isin(constants.PRE_PROCESSES + ["None"]) & data.classifier.isin(constants.CLASSIFIERS)]

wins = {pproc:0 for pproc in ["None"] + constants.PRE_PROCESSES}
for dataset in models.name.unique():
    result_dataset = data.query("name == '{}'".format(dataset))
    max_result = result_dataset[result_dataset[SCORE] == result_dataset[SCORE].max()]
    # Note that results can be similar, so a dataset is included multiple times
    for indx, result in max_result.iterrows():
        wins[result.preprocesses] += 1
default_max_baseline = max(wins, key = lambda key: wins[key])
print("Default PP is:", default_max_baseline)

if not os.path.exists("analysis/plots"):
    os.makedirs("analysis/plots")
if not os.path.exists("analysis/plots/preproc_gain"):
    os.makedirs("analysis/plots/preproc_gain")

mean_scores = []
std_scores = []
for score in constants.CLASSIFIERS_SCORES:
    mean_scores.append(score + "_mean")
    std_scores.append(score + "_std")

REP = 25

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

all_results = []
for rep in range(REP):
    rand_state = np.random.randint(low = 1, high = 1000)
    np.random.seed(rand_state)
    divideFold = KFold(10, random_state = rand_state, shuffle = True)

    reg_models = {}
    reg_models["neural_network"] = lambda: neural_network.MLPRegressor()
    reg_models["ridge"] = lambda: linear_model.Ridge()
    reg_models["gradient_descent"] = lambda: linear_model.SGDRegressor()
    reg_models["svm"] = lambda: svm.SVR(gamma = "auto")
    reg_models["knn"] = lambda: neighbors.KNeighborsRegressor(weights = "distance")
    reg_models["random_forest"] = lambda: ensemble.RandomForestRegressor(random_state = rand_state)
    reg_models["gaussian_process"] = lambda: gaussian_process.GaussianProcessRegressor()
    reg_models["decision_tree"] = lambda: tree.DecisionTreeRegressor(random_state = rand_state)
    reg_models["random"] = lambda: Random()
    reg_models["default"] = lambda: Default()

    results = {}
    for baseline in ["default", "random"]:
        results[baseline] = {}
        for regressor_type in constants.REGRESSORS[:-2]:
            # results[baseline][regressor_type] = {}
            kfold = 0
            results[baseline][regressor_type] = []
            for train_indx, test_indx in divideFold.split(datasets):
                # results[baseline][regressor_type][kfold] = []
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

                    # PREDICTION
                    max_predicted = max(reg_results.keys(), key=(lambda key: reg_results[key]))
                    pp_pred, clf_pred = max_predicted.split("+")
                    predicted_dataset = dataset_info[dataset_info["preprocesses"] == pp_pred]
                    max_pred_pp = predicted_dataset[predicted_dataset[SCORE] == predicted_dataset[SCORE].max()].max()[SCORE]

                    # BASELINE
                    max_baseline = max(baseline_results, key = (lambda key: baseline_results[key]))
                    if baseline == "default":
                        pp_base = default_max_baseline
                    else:
                        pp_base, clf_base = max_baseline.split("+")
                    baseline_dataset = dataset_info[dataset_info["preprocesses"] == pp_base]
                    max_base_pp = baseline_dataset[baseline_dataset[SCORE] == baseline_dataset[SCORE].max()].max()[SCORE]

                    results[baseline][regressor_type].append(max_pred_pp - max_base_pp)
                    print("------------------{}----------------------".format(regressor_type))
                    print("Predicted Score : {}, True Score: {}, PP: {}".format(max(reg_results.values()), float(max_pred_pp), pp_pred))
                    print("Baseline Score : {}, True Score: {}, PP: {}".format(max(baseline_results.values()), float(max_base_pp), pp_base))
                    print("True Max Score : {}, PP: {}".format(float(true_max[SCORE].max()), true_max["preprocesses"]))
                    print("Diff = {}".format(max_pred_pp - max_base_pp))
                    print("----------------------------------------")
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
                true_max = dataset_info[dataset_info[SCORE] == dataset_info[SCORE].max()].iloc[0][SCORE]
                baseline_results = {}
                for model in models:
                    baseline_results[model] = baseline_models[model].predict(meta_data)
                max_baseline = max(baseline_results, key = (lambda key: baseline_results[key]))
                if baseline == "default":
                    pp_base = default_max_baseline
                else:
                    pp_base, clf_base = max_baseline.split("+")
                baseline_dataset = dataset_info[dataset_info["preprocesses"] == pp_base]
                max_base_pp = baseline_dataset[baseline_dataset[SCORE] == baseline_dataset[SCORE].max()].max()[SCORE]
                results[baseline]["true_max"].append(float(true_max) - max_base_pp)

    with open("analysis/plots/preproc_gain/" + SCORE + "_normalized_rep" + str(REP) + "_res_" + str(rep) + ".json", "w") as fd:
        json.dump(results, fd, indent = 4)

    for baseline in results.keys():
        max_val = np.sum(results[baseline]["true_max"])
        del results[baseline]["true_max"]
        for reg in results[baseline]:
            results[baseline][reg] = np.sum(results[baseline][reg])
            results[baseline][reg] /= max_val
            results[baseline][reg] *= 100

    all_results.append(results)

with open("analysis/plots/preproc_gain/" + SCORE + "_normalized_rep_" + str(REP) + ".json", "w") as fd:
    json.dump(all_results, fd, indent = 4)

# def histogram(baseline = 'default'):
#     # fig = plt.figure(figsize = (12, 4))
#     fig = plt.figure()
#     fig.suptitle(baseline, fontsize = 12, fontweight = 'bold')
#     ax = fig.add_subplot(111)
#     ax.bar(results[baseline].keys(), [np.sum(values) for values in results[baseline].values()])
#     plt.xlabel("Regressor", fontsize = 12, fontweight = 'bold')
#     plt.ylabel("PP Gain", fontsize = 12, fontweight = 'bold')
#     plt.xticks(rotation=90)
#     plt.ylim([-4, 5])
#     plt.grid(True, alpha = 0.5, linestyle = 'dotted')
#     plt.gcf().subplots_adjust(bottom=0.60)
#
# for baseline in results.keys():
#     histogram(baseline)
#     plt.savefig("analysis/plots/preproc_gain/" + baseline + ".png", dpi = 100)