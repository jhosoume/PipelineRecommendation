import os
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt

import plotly.io as pio
import plotly.express as px
import plotly.graph_objects as go

import constants
from meta_db.db.DBHelper import DBHelper

pio.templates.default = "plotly_white"

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


grey_palette = ['rgb(208, 209, 211)',
                'rgb(185, 191, 193)',
                'rgb(137, 149, 147)',
                'rgb(44, 54, 60)',
                'rgb(3, 3, 3)'
               ]

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


def histogram(score = "balanced_accuracy_mean"):
    pp_clf_count = {"{}+{}".format(comb.preprocesses, comb.classifier):0 for indx, comb in combinations.iterrows()}
    count = {}
    invert_count = {}
    classifiers = []

    for clf in models.classifier.unique():
        count[clf] = {comb.preprocesses:0 for indx, comb in combinations.iterrows()}
        count[clf]["None"] = 0
        pp_clf_count["None+{}".format(clf)] = 0
        classifiers.append(clf)

    for pp in count[clf].keys():
        invert_count[pp] = {clf:0 for clf in count.keys()}

    mult = 0
    for dataset in models.name.unique():
        result_dataset = data.query("name == '{}'".format(dataset))
        max_result = result_dataset[result_dataset[score] == result_dataset[score].max()]
        if len(max_result) > 1:
            mult += len(max_result) - 1
        # Note that results can be similar, so a dataset is included multiple times
        for indx, result in max_result.iterrows():
            pp_clf_count["{}+{}".format(result.preprocesses, result.classifier)] += 1
            count[result.classifier][result.preprocesses] += 1
            invert_count[result.preprocesses][result.classifier] += 1

    # fig = plt.figure(figsize = (12, 4))
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.bar(pp_clf_count.keys(), pp_clf_count.values())
    # plt.xlabel("PreProcesses + Classifier", fontsize = 12, fontweight = 'bold')
    # plt.ylabel(score.replace("_", " ").capitalize(), fontsize = 12, fontweight = 'bold')
    # plt.xticks(rotation=90)
    # plt.grid(True, alpha = 0.5, linestyle = 'dotted')
    # plt.gcf().subplots_adjust(bottom=0.60)
    combination_df = pd.DataFrame(pp_clf_count, index = [0])
    data_plot = []
    # for clf in count.keys():
    #     bar = go.Bar(
    #         name = clf,
    #         x = list(count[clf].keys()),
    #         y = list(count[clf].values())
    #     )
    #     data_plot.append(bar)
    for indx, pp in enumerate(invert_count.keys()):
        bar = go.Bar(
            name = pp,
            x = list(map(lambda clf: translator[clf], invert_count[pp].keys())),
            y = list(invert_count[pp].values()),
            marker_color = grey_palette[indx]
        )
        data_plot.append(bar)
    fig = go.Figure(data = data_plot)
    fig.update_layout(barmode = 'group')
    fig.update_layout(
        xaxis_title = "Classifier",
        yaxis_title = "Number of Wins",
        uniformtext_minsize = 16,
        font = dict(
            size = 16,
            color = "black"
        )
    )

    fig.update_yaxes(
        showgrid = True,
        linewidth = 1,
        linecolor = "black",
        ticks = "inside",
        mirror = True,
        range = [-1, 70],
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
                       y = 1.15,
                       traceorder= "normal",
                       # bordercolor= "Black",
                       # borderwidth= 0.5
        )
    )
    # fig = px.bar(x = list(pp_clf_count.keys()), y = list(pp_clf_count.values()))
    fig.write_image("analysis/plots/winnings/" + score + ".eps")
    fig.write_image("analysis/plots/winnings/" + score + ".png")
    fig.write_image("/home/jhosoume/unb/tcc/ICDM/img/winnings/" + score + ".eps")
    fig.write_image("/home/jhosoume/unb/tcc/ICDM/img/winnings/" + score + ".png")


for score in constants.CLASSIFIERS_SCORES:
    histogram(score = score + "_mean")
    # plt.savefig("analysis/plots/winnings/" + score + ".png", dpi = 100)
