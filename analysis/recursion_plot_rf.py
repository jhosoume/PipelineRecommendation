import os
from os.path import isfile
import pandas as pd
import numpy as np
import json
from scipy import stats
# import matplotlib.pyplot as plt

import plotly.io as pio
import plotly.express as px
import plotly.graph_objects as go

import constants
from meta_db.db.DBHelper import DBHelper

pio.templates.default = "plotly_white"

db = DBHelper()

SCORE = "balanced_accuracy_mean"
PATH = "analysis/plots/recursion/randomForest/"

# Por regressor
# Número de hits absoluto

translator = {
    "svm": "SVM",
    "logistic_regression": "LG",
    "linear_discriminant": "LD",
    "kneighbors": "kNN",
    "decision_tree": "DT",
    "gaussian_nb": "GNB",
    "random_forest": "RF",
    "randomForest": "RF",
    "gradient_boosting": "GB",
    "neural_network": "NN",
    "knn": "kNN",
    "dwnn": "DWNN",
    "Svm": "SVM",
    "ann": "ANN",
    "cart": "CART",
    "random": "Random",
    "default": "Default"
}

translator_res = {
    "pp_wins": "Pre-Processor Hit",
    "clf_wins": "Classifier Hit",
    "wins": "Hit",
}


grey_palette = ['rgb(208, 209, 211)',
                'rgb(185, 191, 193)',
                'rgb(137, 149, 147)',
                'rgb(44, 54, 60)',
                'rgb(3, 3, 3)'
               ]

totals = []
results = []
for filename in os.listdir(PATH):
    if isfile(PATH + filename):
        if "numdatasets" in filename:
            with open(PATH + filename, "r") as fd:
                totals.append(json.load(fd))
        else:
            with open(PATH + filename, "r") as fd:
                try:
                    results.append(json.load(fd))
                except:
                    import pdb; pdb.set_trace()

mean_res = {type_res: list(np.sum([res[type_res]["randomForest"]
                for res in results], axis = 0)/len(results))
                for type_res in results[0].keys()}
std_res = {type_res: list(np.std([res[type_res]["randomForest"]
                for res in results], axis = 0)/len(results))
                for type_res in results[0].keys()}
mean_totes = list(np.sum([tot["randomForest"]
                for tot in totals], axis = 0)/len(totals))
std_totes = list(np.std([tot["randomForest"]
                for tot in totals], axis = 0)/len(totals))

if not os.path.exists("analysis/plots"):
    os.makedirs("analysis/plots")
if not os.path.exists("analysis/plots/recursion"):
    os.makedirs("analysis/plots/recursion")

data_plot = []
for indx, res in enumerate(mean_res.keys())  :
    bar = go.Bar(
        name = translator_res[res],
        x = ["Round {}".format(round + 1) for round in range(len(mean_res[res]))],
        y = mean_res[res],
        error_y = dict(
            type = "data",
            array = std_res[res]),
        marker_color = grey_palette[indx]
    )
    data_plot.append(bar)
scatter = go.Scatter(
    name = "Max",
    x = ["Round {}".format(round + 1) for round in range(len(mean_totes))],
    y = mean_totes,
    error_y = dict(
        type = "data",
        array = std_totes),
    marker_color = grey_palette[-1],
    mode = "markers"

)
data_plot.append(scatter)
fig = go.Figure(data = data_plot)
fig.update_layout(barmode = 'group')
fig.update_layout(
    xaxis_title = "RandomForest",
    yaxis_title = "Number of Hits",
    uniformtext_minsize = 16,
    font = dict(
        size = 16,
        color = "black"
    ),
    yaxis_title_standoff = 0

)

fig.update_yaxes(
    showgrid = True,
    linewidth = 1,
    linecolor = "black",
    ticks = "inside",
    mirror = True,
    range = [0, 42],
    tickfont= dict(
        size= 16,
        color = 'black'
    ),
    titlefont = dict(
        size = 18
    )
)

fig.update_xaxes(
    showgrid = True,
    linewidth = 1,
    linecolor = "black",
    ticks = "inside",
    tickson = "boundaries",
    mirror = True,
    tickfont= dict(
        size= 16,
        color = 'black'
    ),
    titlefont = dict(
        size = 18
    )
)

fig.update_yaxes(
    zeroline = True,
    zerolinewidth = 2,
    zerolinecolor = "black",
)

fig.update_layout(legend_orientation="h")
fig.update_layout(
    legend = dict(
                   x = 0,
                   y = 1.12,
                   traceorder= "normal",
                   # bordercolor= "Black",
                   # borderwidth= 0.5
    )
)

# fig = px.bar(x = list(pp_clf_count.keys()), y = list(pp_clf_count.values()))
fig.write_image("analysis/plots/recursion/rf.eps")
fig.write_image("analysis/plots/recursion/rf.png")
fig.write_image("/home/jhosoume/unb/tcc/ICDM/img/recursion_analysis/rec.rf.eps")
fig.write_image("/home/jhosoume/unb/tcc/ICDM/img/recursion_analysis/rec.rf.png")
