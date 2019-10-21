import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import constants
from meta_db.db.DBHelper import DBHelper

db = DBHelper()

regressors = pd.DataFrame(db.get_all_regressors(), columns = db.regressor_columns()).drop("id", axis = 1)

if not os.path.exists("analysis/plots"):
    os.makedirs("analysis/plots")
if not os.path.exists("analysis/plots/meta_level"):
    os.makedirs("analysis/plots/meta_level")

# def box_plot(score = "accuracy_mean"):
score = "accuracy"
regressor_score = "neg_mean_absolute_error_mean"
fig = plt.figure(figsize = (24, 8))
for clf in constants.CLASSIFIERS:
    data = []
    ax = fig.add_subplot(len(constants.CLASSIFIERS), 1, 1)
    clf_data = regressors.query("classifier == '{}' and score == '{}'".format(clf, score))
    for reg in constants.REGRESSORS:
        reg_info = clf_data.query("name == '{}'".format(reg))[regressor_score]
        data.append(reg_info * -1)
        ax.boxplot(data, showmeans = True, meanline = True,
                    labels = [name.replace("_", " ").capitalize() for name in constants.REGRESSORS])
        plt.xlabel("Classifier", fontsize = 12, fontweight = 'bold')
        plt.ylabel("Mean Absolute Error", fontsize = 12, fontweight = 'bold')
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.grid(True, alpha = 0.5, linestyle = 'dotted')
#
# for score in constants.CLASSIFIERS_SCORES:
#     box_plot(score = score + "_mean")
#     plt.savefig("analysis/plots/meta_base/" + score + ".png", dpi = 100)
