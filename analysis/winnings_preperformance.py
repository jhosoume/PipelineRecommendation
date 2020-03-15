import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import constants
from meta_db.db.DBHelper import DBHelper

db = DBHelper()

models = pd.DataFrame(db.get_all_models(), columns = db.models_columns()).drop("id", axis = 1)
combinations = pd.DataFrame(db.get_all_combinations(), columns = db.combinations_columns())
preperformance = pd.DataFrame(db.get_all_preperformance(), columns = db.preperformance_columns()).drop("id", axis = 1)
# Not null preperformance
preperformance = preperformance[~preperformance.isnull().any(axis = 1)]
preperformance = pd.merge(preperformance, combinations, left_on = "combination_id", right_on = "id").drop(["combination_id", "id", "num_preprocesses"], axis = 1)

models = models.rename(columns = {"model": "classifier"})
models["preprocesses"] = "None"
data = pd.concat([models, preperformance], sort = False)
data = data[data.classifier != "neural_network"]
models = models[models.classifier != "neural_network"]

if not os.path.exists("analysis/plots"):
    os.makedirs("analysis/plots")
if not os.path.exists("analysis/plots/winnings"):
    os.makedirs("analysis/plots/winnings")


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
