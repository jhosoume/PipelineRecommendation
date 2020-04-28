import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import plotly.io as pio
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff

import constants
from meta_db.db.DBHelper import DBHelper

db = DBHelper()

regressors = pd.DataFrame(db.get_all_regressors_preperformance(), columns = ["name", "score", "max_error", "mean_absolute_error", "mean_squared_error", "r2_score", "median_absolute_error", "classifier", "preprocesses"] )
regressors_nopp = pd.DataFrame(db.get_all_regressors(), columns = db.regressor_columns()).drop("id", axis = 1)

if not os.path.exists("analysis/plots"):
    os.makedirs("analysis/plots")
if not os.path.exists("analysis/plots/meta_preperformance"):
    os.makedirs("analysis/plots/meta_preperformance")

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

x_axis = [translator[reg] for reg in constants.REGRESSORS]
y_axis = ["{}+{}".format(pp, translator[clf]) for pp in (["None"] + constants.PRE_PROCESSES) for clf in constants.CLASSIFIERS]

pp_clfs =  [(pp, clf)  for pp in (["None"] + constants.PRE_PROCESSES) for clf in constants.CLASSIFIERS]

score = "accuracy"
regressor_score = "mean_squared_error"

data = np.zeros((len(y_axis), len(x_axis)))
data_std = np.zeros((len(y_axis), len(x_axis)))

for y_index, (pproc, clf) in enumerate(pp_clfs):
    regs = regressors_nopp if pproc == "None" else regressors
    if pproc == "None":
        clf_data = regs.query("classifier == '{}' and score == '{}'".format(clf, score))
    else:
        clf_data = regs.query("classifier == '{}' and score == '{}' and preprocesses == '{}'".format(clf, score, pproc))
    for x_index, reg in enumerate(constants.REGRESSORS):
        reg_info = clf_data.query("name == '{}'".format(reg))[regressor_score]
        data[y_index, x_index] = np.mean(reg_info)
        data_std[y_index, x_index] = np.std(reg_info)

text = []
for line in range(len(y_axis)):
    line_text = []
    for col in range(len(x_axis)):
        line_text.append( "{:1.4f} {} {:1.4f}".format(
                                    np.around(data[line, col], decimals = 4),
                                    u'\xb1',
                                    np.around(data_std[line, col], decimals = 4) ) )
    text.append(line_text)


fig = ff.create_annotated_heatmap(data.tolist(), annotation_text = text,
                                  x = x_axis, y = y_axis,
                                  colorscale='Greys', hoverinfo='z')

fig.show()
fig.write_image("analysis/plots/meta_preperformance/" + score + "_heatmap_ppordered.eps",
                width = 1000, height = 1200, scale = 1)


# fig = plt.figure(figsize = (24, 24))
# fig.suptitle(score, fontsize = 12, fontweight = 'bold')
# for indx, clf in enumerate(constants.CLASSIFIERS):
#     data = []
#     ax = fig.add_subplot(len(constants.CLASSIFIERS), 1, indx + 1)
#     ax.set_title("{} {}".format(clf, preprocess))
#     clf_data = regressors.query("classifier == '{}' and score == '{}' and preprocesses == '{}'".format(clf, score, preprocess))
#     for reg in constants.REGRESSORS:
#         reg_info = clf_data.query("name == '{}'".format(reg))[regressor_score]
#         data.append(reg_info)
#     ax.boxplot(data, showmeans = True, meanline = True, showfliers = False,
#                 labels = [name.replace("_", " ").capitalize() for name in constants.REGRESSORS])
#     plt.ylabel("Mean Squared Error", fontsize = 12, fontweight = 'bold')
#     plt.ylim([0.0, 0.5])
#     plt.xlabel("Regressor", fontsize = 12, fontweight = 'bold')
#     plt.tight_layout()
#     plt.grid(True, alpha = 0.5, linestyle = 'dotted')
#     for axes in fig.get_axes():
#         axes.label_outer()
# plt.savefig("analysis/plots/meta_preperformance/mse_accuracy_{}.png".format(preprocess), dpi = 100)
#
# for preprocess in constants.PRE_PROCESSES:
#      box_plot(preprocess)
