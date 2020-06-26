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

pio.templates.default = "plotly_white"

db = DBHelper()

regressors = pd.DataFrame(db.get_all_regressors_preperformance(), columns = ["name", "score", "max_error", "mean_absolute_error", "mean_squared_error", "r2_score", "median_absolute_error", "classifier", "preprocesses"] )
regressors_nopp = pd.DataFrame(db.get_all_regressors(), columns = db.regressor_columns()).drop("id", axis = 1)

regressors_nopp["preprocesses"] = "None"
all_regs = pd.concat([regressors, regressors_nopp])
all_regs = all_regs[all_regs.name.isin(constants.REGRESSORS)]

if not os.path.exists("analysis/plots"):
    os.makedirs("analysis/plots")
if not os.path.exists("analysis/plots/meta_preperformance"):
    os.makedirs("analysis/plots/meta_preperformance")
if not os.path.exists("analysis/plots/meta_preperformance/clf_group_csv"):
    os.makedirs("analysis/plots/meta_preperformance/clf_group_csv")
if not os.path.exists("analysis/plots/meta_preperformance/pp_group_csv"):
    os.makedirs("analysis/plots/meta_preperformance/pp_group_csv")
if not os.path.exists("analysis/plots/meta_preperformance/all_group_csv"):
    os.makedirs("analysis/plots/meta_preperformance/all_group_csv")
if not os.path.exists("analysis/plots/meta_preperformance/clf_group"):
    os.makedirs("analysis/plots/meta_preperformance/clf_group")
if not os.path.exists("analysis/plots/meta_preperformance/pp_group"):
    os.makedirs("analysis/plots/meta_preperformance/pp_group")
if not os.path.exists("analysis/plots/meta_preperformance/all_group"):
    os.makedirs("analysis/plots/meta_preperformance/all_group")

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

score = "balanced_accuracy"
regressor_score = "mean_squared_error"


data_regs = all_regs[all_regs.preprocesses.isin(constants.PRE_PROCESSES + ["None"])]
for clf in constants.CLASSIFIERS:
    clf_data = data_regs.query("classifier == '{}' and score == '{}'".format(clf, score))
    clf_data = clf_data[["name", regressor_score]]
    clf_data.name = clf_data.name.apply(lambda reg: translator[reg])
    # Sorting
    mapping = {reg: indx for indx, reg in enumerate(constants.REG_ORDER)}
    key = clf_data["name"].map(mapping)
    clf_data = clf_data.iloc[key.argsort()]
    data = {reg:list(clf_data[clf_data.name == reg].mean_squared_error) for reg in clf_data.name.unique()}
    dframe = pd.DataFrame(data)
    dframe.to_csv("analysis/plots/meta_preperformance/clf_group_csv/{}.csv".format(clf), index = False, sep = "\t")

    boxp = px.box(clf_data, x="name", y="mean_squared_error", color_discrete_sequence = ["grey"])

    fig = go.Figure(data = boxp)
    fig.update_layout(
        title = translator[clf],
        xaxis_title = "Regressor",
        yaxis_title = "Mean Square Error",
        uniformtext_minsize = 16
    )

    fig.update_yaxes(
        showgrid = True,
        linewidth = 1,
        linecolor = "black",
        ticks = "inside",
        mirror = True,
        range = [0, 0.15],
        tickfont= dict(
            size= 18,
            color = 'black'
        ),
        titlefont = dict(
            size = 20
        )
    )

    fig.update_xaxes(
        showgrid = True,
        linewidth = 1,
        linecolor = "black",
        ticks = "inside",
        tickson = "boundaries",
        mirror = True,
        tickfont = dict(
            size = 18,
            color = 'black'
        ),
        titlefont = dict(
            size = 20
        )
    )

    fig.update_yaxes(
        zeroline = True,
        zerolinewidth = 1.1,
        zerolinecolor = "black",
    )
    fig.write_image("/home/jhosoume/unb/tcc/ICDM/img/meta_level_analysis/{}_{}.eps".format(clf, score))
    fig.write_image("analysis/plots/meta_preperformance/clf_group/{}_{}.eps".format(clf, score))
    fig.write_image("/home/jhosoume/unb/tcc/ICDM/img/meta_level_analysis/{}_{}.png".format(clf, score))
    fig.write_image("analysis/plots/meta_preperformance/clf_group/{}_{}.png".format(clf, score))
    # fig.show()

data_regs = all_regs[all_regs.classifier.isin(constants.CLASSIFIERS)]
for pp in (["None"] + constants.PRE_PROCESSES):
    pp_data = data_regs.query("preprocesses == '{}' and score == '{}'".format(pp, score))
    pp_data = pp_data[["name", regressor_score]]
    pp_data.name = pp_data.name.apply(lambda reg: translator[reg])
    mapping = {reg: indx for indx, reg in enumerate(constants.REG_ORDER)}
    key = pp_data["name"].map(mapping)
    pp_data = pp_data.iloc[key.argsort()]
    # pp_data = pp_data.sort_values(by = ["name"])
    pp_data.to_csv("analysis/plots/meta_preperformance/pp_group_csv/{}.csv".format(pp), index = False)

    data = {reg:list(pp_data[pp_data.name == reg].mean_squared_error) for reg in pp_data.name.unique()}
    dframe = pd.DataFrame(data)
    dframe.to_csv("analysis/plots/meta_preperformance/pp_group_csv/{}.csv".format(pp), index = False, sep = "\t")

    boxp = px.box(pp_data, x="name", y="mean_squared_error", color_discrete_sequence = ["grey"])

    fig = go.Figure(data = boxp)
    fig.update_layout(
        title = pp,
        xaxis_title = "Regressor",
        yaxis_title = "Mean Square Error"
    )

    fig.update_yaxes(
        showgrid = True,
        linewidth = 1,
        linecolor = "black",
        ticks = "inside",
        mirror = True,
        range = [-0.005, 0.15]
    )

    fig.update_xaxes(
        showgrid = True,
        linewidth = 1,
        linecolor = "black",
        ticks = "inside",
        tickson = "boundaries",
        mirror = True
    )

    fig.update_yaxes(
        zeroline = True,
        zerolinewidth = 1.3,
        zerolinecolor = "black",
    )
    fig.write_image("analysis/plots/meta_preperformance/pp_group/{}_{}.eps".format(pp, score))
    fig.write_image("analysis/plots/meta_preperformance/pp_group/{}_{}.png".format(pp, score))
    # fig.show()

data_regs = all_regs[
                all_regs.preprocesses.isin(constants.PRE_PROCESSES + ["None"]) &
                all_regs.classifier.isin(constants.CLASSIFIERS)]
all_data = data_regs.query("score == '{}'".format(score))
all_data = all_data[["name", regressor_score]]
all_data.name = all_data.name.apply(lambda reg: translator[reg])

mapping = {reg: indx for indx, reg in enumerate(constants.REG_ORDER)}
key = all_data["name"].map(mapping)
all_data = all_data.iloc[key.argsort()]

data = {reg:list(all_data[all_data.name == reg].mean_squared_error) for reg in all_data.name.unique()}
dframe = pd.DataFrame(data)
dframe.to_csv("analysis/plots/meta_preperformance/all_group_csv/all.csv", index = False, sep = "\t")
boxp = px.box(all_data, x="name", y="mean_squared_error", color_discrete_sequence = ["grey"])

fig = go.Figure(data = boxp)
fig.update_layout(
    xaxis_title = "Regressor",
    yaxis_title = "Mean Square Error"
)

fig.update_yaxes(
    showgrid = True,
    linewidth = 1,
    linecolor = "black",
    ticks = "inside",
    mirror = True,
    range = [-0.005, 0.15]
)

fig.update_xaxes(
    showgrid = True,
    linewidth = 1,
    linecolor = "black",
    ticks = "inside",
    tickson = "boundaries",
    mirror = True
)

fig.update_yaxes(
    zeroline = True,
    zerolinewidth = 1.3,
    zerolinecolor = "black",
)
fig.write_image("analysis/plots/meta_preperformance/all_group/{}.eps".format(score))
fig.write_image("analysis/plots/meta_preperformance/all_group/{}.png".format(score))
# fig.show()
# data = np.zeros((len(y_axis), len(x_axis)))
# data_std = np.zeros((len(y_axis), len(x_axis)))
#
#
# for y_index, (pproc, clf) in enumerate(pp_clfs):
#     regs = regressors_nopp if pproc == "None" else regressors
#     if pproc == "None":
#         clf_data = regs.query("classifier == '{}' and score == '{}'".format(clf, score))
#     else:
#         clf_data = regs.query("classifier == '{}' and score == '{}' and preprocesses == '{}'".format(clf, score, pproc))
#     for x_index, reg in enumerate(constants.REGRESSORS):
#         reg_info = clf_data.query("name == '{}'".format(reg))[regressor_score]
#         data[y_index, x_index] = np.mean(reg_info)
#         data_std[y_index, x_index] = np.std(reg_info)

# text = []
# for line in range(len(y_axis)):
#     line_text = []
#     for col in range(len(x_axis)):
#         line_text.append( "{:1.4f} {} {:1.4f}".format(
#                                     np.around(data[line, col], decimals = 4),
#                                     u'\xb1',
#                                     np.around(data_std[line, col], decimals = 4) ) )
#     text.append(line_text)
#
#
# fig = ff.create_annotated_heatmap(data.tolist(), annotation_text = text,
#                                   x = x_axis, y = y_axis,
#                                   colorscale='Greys', hoverinfo='z')
#
# fig.show()
# fig.write_image("analysis/plots/meta_preperformance/" + score + "_heatmap_ppordered.png",
#                 width = 1000, height = 1200, scale = 1)
#
#
