import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import constants
from meta_db.db.DBHelper import DBHelper

db = DBHelper()

combinations = pd.DataFrame(db.get_all_combinations(), columns = db.combinations_columns())
preperformance = pd.DataFrame(db.get_all_preperformance(), columns = db.preperformance_columns()).drop("id", axis = 1)

if not os.path.exists("analysis/plots"):
    os.makedirs("analysis/plots")
if not os.path.exists("analysis/plots/preperformance"):
    os.makedirs("analysis/plots/preperformance")

def box_plot(score = "accuracy_mean", preprocess = "SMOTE"):
    data = []
    for model in constants.CLASSIFIERS:
        combination_id = int(combinations[combinations.preprocesses == preprocess][combinations.classifier == model]["id"])
        data.append(preperformance[preperformance.combination_id == combination_id][score].dropna())
    # import pdb; pdb.set_trace()
    fig = plt.figure(figsize = (12, 4))
    ax = fig.add_subplot(111)
    ax.boxplot(data, showmeans = True, meanline = True,
                labels = [name.replace("_", " ").capitalize() for name in constants.CLASSIFIERS]
               )
    plt.xlabel("Classifier", fontsize = 12, fontweight = 'bold')
    plt.ylabel(score.replace("_", " ").capitalize(), fontsize = 12, fontweight = 'bold')
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.grid(True, alpha = 0.5, linestyle = 'dotted')

for score in constants.CLASSIFIERS_SCORES:
    for pre_proc in constants.PRE_PROCESSES:
# score = "accuracy"
# pre_proc = "ADASYN"
        box_plot(score = score + "_mean", preprocess = pre_proc)
        plt.savefig("analysis/plots/preperformance/" + score + "_" + pre_proc + ".png", dpi = 100)
