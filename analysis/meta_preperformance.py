import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import constants
from meta_db.db.DBHelper import DBHelper

db = DBHelper()

regressors = pd.DataFrame(db.get_all_regressors_preperformance(), columns = ["name", "score", "max_error", "mean_absolute_error", "r2_score", "median_absolute_error", "classifier", "preprocesses"] )

if not os.path.exists("analysis/plots"):
    os.makedirs("analysis/plots")
if not os.path.exists("analysis/plots/meta_preperformance"):
    os.makedirs("analysis/plots/meta_preperformance")

def box_plot(preprocess):
    score = "accuracy"
    regressor_score = "mean_absolute_error"
    fig = plt.figure(figsize = (24, 24))
    fig.suptitle(score, fontsize = 12, fontweight = 'bold')
    for indx, clf in enumerate(constants.CLASSIFIERS):
        data = []
        ax = fig.add_subplot(len(constants.CLASSIFIERS), 1, indx + 1)
        ax.set_title("{} {}".format(clf, preprocess))
        clf_data = regressors.query("classifier == '{}' and score == '{}' and preprocesses == '{}'".format(clf, score, preprocess))
        for reg in constants.REGRESSORS:
            reg_info = clf_data.query("name == '{}'".format(reg))[regressor_score]
            data.append(reg_info)
        ax.boxplot(data, showmeans = True, meanline = True, showfliers = False,
                    labels = [name.replace("_", " ").capitalize() for name in constants.REGRESSORS])
        plt.ylabel("Mean Squared Error", fontsize = 12, fontweight = 'bold')
        plt.ylim([0.0, 0.5])
        plt.xlabel("Regressor", fontsize = 12, fontweight = 'bold')
        plt.tight_layout()
        plt.grid(True, alpha = 0.5, linestyle = 'dotted')
        for axes in fig.get_axes():
            axes.label_outer()
    plt.savefig("analysis/plots/meta_preperformance/mse_accuracy_{}.png".format(preprocess), dpi = 100)

for preprocess in constants.PRE_PROCESSES:
     box_plot(preprocess)
