import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import constants
from meta_db.db.DBHelper import DBHelper

db = DBHelper()

models = pd.DataFrame(db.get_all_models(), columns = db.models_columns()).drop("id", axis = 1)

if not os.path.exists("analysis/plots"):
    os.makedirs("analysis/plots")
if not os.path.exists("analysis/plots/meta_base"):
    os.makedirs("analysis/plots/meta_base")

def box_plot(score = "accuracy_mean"):
    data = []
    for model in constants.CLASSIFIERS:
        data.append(models[models.model == model][score])
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
    box_plot(score = score + "_mean")
    plt.savefig("analysis/plots/meta_base/" + score + ".png", dpi = 100)
