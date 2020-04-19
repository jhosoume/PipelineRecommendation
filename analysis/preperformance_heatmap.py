import os
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt

import plotly.io as pio
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff

import constants
from meta_db.db.DBHelper import DBHelper

db = DBHelper()

translator = {
    "svm": "SVM",
    "logistic_regression": "LG",
    "linear_discriminant": "LD",
    "kneighbors": "kNN",
    "decision_tree": "DT",
    "gaussian_nb": "GNB",
    "random_forest": "RF",
    "gradient_boosting": "GB"

}

models = pd.DataFrame(db.get_all_models(), columns = db.models_columns()).drop("id", axis = 1)
combinations = pd.DataFrame(db.get_all_combinations(), columns = db.combinations_columns())
preperformance = pd.DataFrame(db.get_all_preperformance(), columns = db.preperformance_columns()).drop("id", axis = 1)

if not os.path.exists("analysis/plots"):
    os.makedirs("analysis/plots")
if not os.path.exists("analysis/plots/preperformance"):
    os.makedirs("analysis/plots/preperformance")

score = "accuracy_mean"

y_axis = list(map(lambda clf: translator[clf], constants.CLASSIFIERS))
x_axis = ['None'] + constants.PRE_PROCESSES

data = np.zeros((len(x_axis), len(y_axis)))
data_std = np.zeros((len(x_axis), len(y_axis)))

for y_index, clf in enumerate(constants.CLASSIFIERS):
    data[0, y_index] = np.mean(models[models.model == clf][score])
    data_std[0, y_index] = np.std(models[models.model == clf][score])
    for x_index, pre_proc in enumerate(constants.PRE_PROCESSES):
        combination_id = int(combinations[combinations.preprocesses == pre_proc][combinations.classifier == clf]["id"])
        data[x_index + 1, y_index] = np.mean(preperformance[preperformance.combination_id == combination_id][score].dropna())
        data_std[x_index + 1, y_index] = np.std(preperformance[preperformance.combination_id == combination_id][score].dropna())

text = []
for line in range(len(x_axis)):
    line_text = []
    for col in range(len(y_axis)):
        line_text.append( "{} {} {}".format(
                                    np.around(data[line, col], decimals = 2),
                                    u'\xb1',
                                    np.around(data_std[line, col], decimals = 2) ) )
    text.append(line_text)


fig = ff.create_annotated_heatmap(data.tolist(), annotation_text = text,
                                  y = x_axis, x = y_axis,
                                  colorscale='Greys', hoverinfo='z')

fig.show()
fig.write_image("analysis/plots/preperformance/" + score + "_heatmap.png")