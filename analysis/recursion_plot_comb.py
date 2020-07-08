import os
import pandas as pd
import numpy as np
import json
# import matplotlib.pyplot as plt

import plotly.io as pio
import plotly.express as px
import plotly.graph_objects as go

import constants
from meta_db.db.DBHelper import DBHelper

pio.templates.default = "plotly_white"

db = DBHelper()

SCORE = "balanced_accuracy_mean"

# Por regressor
# Número de hits absoluto


translator = {
    "svm": "SVM",
    "logistic_regression": "LG",
    "linear_discriminant": "LD",
    "kneighbors": "kNN",
    "knn": "kNN",
    "decision_tree": "DT",
    "gaussian_nb": "GNB",
    "random_forest": "RF",
    "gradient_boosting": "GB"

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

# OLD
# results = {'pp_wins': {'svm': [3, 3, 3, 3, 3], 'knn': [4, 2, 0, 0, 0], 'random_forest': [5, 3, 3, 3, 3], 'decision_tree': [4, 0, 0, 0, 0]},
#           'clf_wins': {'svm': [3, 3, 3, 3, 3], 'knn': [2, 2, 0, 0, 0], 'random_forest': [6, 3, 3, 3, 3], 'decision_tree': [1, 0, 0, 0, 0]},
#               'wins': {'svm': [2, 3, 3, 3, 3], 'knn': [2, 2, 0, 0, 0], 'random_forest': [3, 3, 3, 3, 3], 'decision_tree': [1, 0, 0, 0, 0]}}

with open("analysis/plots/recursion/{}_combination_numdatasets.json".format(SCORE), "r") as fd:
    totals = json.load(fd)
with open("analysis/plots/recursion/{}_combination.json".format(SCORE), "r") as fd:
    results = json.load(fd)

num_dt = {reg:[0] * len(results["pp_wins"][reg]) for reg in results["pp_wins"].keys()}

# for reg in results["pp_wins"].keys():
#     div = initial
#     for round in range(len(results["pp_wins"][reg])):
#         if div == 0:
#         num_dt[reg][round] = div
#             break
#         store_round = results["pp_wins"][reg][round]
#         results["pp_wins"][reg][round] /= (div/100)
#         div = store_round
#
# for res in list(results.keys())[1:]:
#     for reg in results[res].keys():
#         for round in range(len(results[res][reg])):
#             div = num_dt[reg][round]
#             if div == 0:
#                 break
#             results[res][reg][round] /= (div/100)

if not os.path.exists("analysis/plots"):
    os.makedirs("analysis/plots")
if not os.path.exists("analysis/plots/recursion"):
    os.makedirs("analysis/plots/recursion")

for reg in results["pp_wins"]:
    data_plot = []
    for indx, res in enumerate(results.keys()):
        bar = go.Bar(
            name = translator_res[res],
            x = ["Round {}".format(round + 1) for round in range(len(results[res][reg]))],
            y = results[res][reg],
            marker_color = grey_palette[indx]
        )
        data_plot.append(bar)
    scatter = go.Scatter(
        name = "Max",
        x = ["Round {}".format(round + 1) for round in range(len(totals[reg]))],
        y = totals[reg],
        marker_color = grey_palette[-1],
        mode = "markers"

    )
    data_plot.append(scatter)
    fig = go.Figure(data = data_plot)
    fig.update_layout(barmode = 'group')
    fig.update_layout(
        xaxis_title = "Regressor",
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
    fig.write_image("analysis/plots/recursion/reg.{}.eps".format(reg))
    fig.write_image("analysis/plots/recursion/reg.{}.png".format(reg))
    fig.write_image("/home/jhosoume/unb/tcc/ICDM/img/recursion_analysis/reg.{}.eps".format(reg))
    fig.write_image("/home/jhosoume/unb/tcc/ICDM/img/recursion_analysis/reg.{}.png".format(reg))
