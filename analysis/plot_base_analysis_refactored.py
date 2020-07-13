import os
import pandas as pd
from scipy import stats
import numpy as np

import constants
from meta_db.db.DBHelper import DBHelper

import plotly.io as pio
import plotly.express as px
import plotly.graph_objects as go

SCORE = "balanced_accuracy_mean"
DIST_FUNCTION = stats.sem
REP = 8

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

pio.templates.default = "plotly_white"


grey_palette = ['rgb(208, 209, 211)',
                'rgb(137, 149, 147)',
               ]

db = DBHelper()

if not os.path.exists("analysis/plots"):
    os.makedirs("analysis/plots")
if not os.path.exists("analysis/plots/base_analysis"):
    os.makedirs("analysis/plots/base_analysis")


metadata = pd.DataFrame(db.get_all_metadata(), columns = db.metadata_columns()).drop("id", axis = 1)
models = pd.DataFrame(db.get_all_models(), columns = db.models_columns()).drop("id", axis = 1)
combinations = pd.DataFrame(db.get_all_combinations(), columns = db.combinations_columns())
preperformance = pd.DataFrame(db.get_all_preperformance(), columns = db.preperformance_columns()).drop("id", axis = 1)
# Not null preperformance
preperformance = preperformance[~preperformance.isnull().any(axis = 1)]
preperformance = pd.merge(preperformance, combinations, left_on = "combination_id", right_on = "id").drop(["combination_id", "id", "num_preprocesses"], axis = 1)

models = models.rename(columns = {"model": "classifier"})
models["preprocesses"] = "None"
scores = pd.concat([models, preperformance], sort = False)
scores = scores[scores.preprocesses.isin(constants.PRE_PROCESSES + ["None"]) & scores.classifier.isin(constants.CLASSIFIERS)]
scores = scores.drop(columns = ["recall_micro_mean", "recall_micro_std", "precision_macro_mean", "precision_macro_std", "recall_macro_std", "recall_macro_mean", "accuracy_mean", "accuracy_std","precision_micro_mean", "precision_micro_std", "f1_micro_mean", "f1_micro_std", "f1_macro_std", "f1_macro_mean","fit_time_mean", "fit_time_std", "score_time_mean", "score_time_std" ])

metadata_means = {feature: np.mean(metadata[feature]) for feature in metadata.columns if feature != "name"}
metadata.fillna(value = metadata_means, inplace = True)

data = pd.merge(metadata, scores, on = "name")

data = data[data.preprocesses.isin(constants.PRE_PROCESSES + ["None"]) & data.classifier.isin(constants.CLASSIFIERS)]
combinations_strings = ["{}+{}".format(pp, clf) for pp in ["None"] + constants.PRE_PROCESSES
                                                for clf in constants.CLASSIFIERS]

# Function to get only datasets with all results (combinations)
def filter_dataset(database):
    datasets_filtered = []
    for dataset in database.name.unique():
        split = database[database.name == dataset]
        keep = True
        for clf in constants.CLASSIFIERS:
            for pp in constants.PRE_PROCESSES + ['None']:
                if len(split[(split.classifier == clf) & (split.preprocesses == pp)]) < 1:
                    keep = False
        if keep:
            datasets_filtered.append(dataset)
    return datasets_filtered

datasets = pd.Series(filter_dataset(data))
print("Num datasets:", len(datasets))

# Redimensioning the metabase in data (removing other scores)
list_datasets = []
for dataset in datasets:
    meta_features = metadata[metadata.name == dataset]
    dataset_dict = {col:value for col, value in zip(list(meta_features), list(meta_features.values[0]))}
    for pp in ['None'] + constants.PRE_PROCESSES:
        for clf in constants.CLASSIFIERS:
            dataset_dict["{}+{}".format(pp, clf)] = float(scores[(scores.name == dataset) & (scores.preprocesses == pp) & (scores.classifier == clf)]["balanced_accuracy_mean"])
    list_datasets.append(dataset_dict)

meta_base = pd.DataFrame(list_datasets)

# Calculate real max
true_max = pd.DataFrame()
true_max["dataset"] = meta_base["name"]
true_max["combination"] = meta_base[combinations_strings].idxmax(axis = 1)
true_max["value"] = meta_base[combinations_strings].max(axis = 1)
sum_true_max = true_max["value"].sum()
import pdb; pdb.set_trace()

# CALCULATIONS
all_predictions = []
for rep in range(REP):
    predictions = pd.read_csv("analysis/plots/base_analysis/predictions_{}.csv".format(rep)).drop("Unnamed: 0", axis = 1)
    predictions["max_pred"] = predictions[combinations_strings].idxmax(axis = 1)
    predictions["real_max_pred"] = predictions.apply(lambda pred: float(meta_base[meta_base.name == pred.dataset][pred.max_pred]),axis = 1)
    result = {}
    for baseline in ["random", "default"]:
        result[baseline] = {}
        baseline_sum = predictions[predictions.regressor == baseline]["real_max_pred"].sum()
        for reg in filter(lambda reg: not reg in ["random", "default"], constants.REGRESSORS):
            reg_sum = predictions[predictions.regressor == reg]["real_max_pred"].sum()
            result[baseline][reg] = (reg_sum - baseline_sum) * 100 / (sum_true_max - baseline_sum)
    all_predictions.append(result)

# PLOTTING
results = {}
for baseline in ["random", "default"]:
    results[baseline] = {}
    for reg in filter(lambda reg: not reg in ["random", "default"], constants.REGRESSORS):
        results[baseline][reg] = {}
        results[baseline][reg]["mean"] = np.mean([res[baseline][reg] for res in all_predictions])
        results[baseline][reg]["dist"] = DIST_FUNCTION([res[baseline][reg] for res in all_predictions])

data_plot = []
for indx, baseline in enumerate(results.keys()):
    bar = go.Bar(
        name = translator[baseline],
        x = list(map(lambda reg: translator[reg], results[baseline].keys())),
        y = [results[baseline][reg]["mean"] for reg in results[baseline].keys()],
        error_y = dict(
            type = "data",
            array = [results[baseline][reg]["dist"] for reg in results[baseline].keys()]),
        marker_color = grey_palette[indx],
    )
    data_plot.append(bar)

    # bar = go.Bar(
    #     name = translator[baseline],
    #     x = list(map(lambda reg: translator[reg], non_normalized_results[baseline].keys())),
    #     y = list([np.sum(values) for values in non_normalized_results.values()]),
    #     marker_color = grey_palette[1]
    # )

fig = go.Figure(data = data_plot)

fig.update_layout(barmode = 'group')
fig.update_layout(
    xaxis_title = "Regressor",
    yaxis_title = "Gain (%) ",
    uniformtext_minsize = 16,
    font = dict(
        size = 16,
    ),
    yaxis_title_standoff = 0    ,

)

fig.update_yaxes(
    showgrid = True,
    linewidth = 1,
    linecolor = "black",
    ticks = "inside",
    mirror = True,
    range = [-80, 100],
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
    ),

)

fig.update_yaxes(
    zeroline = True,
    zerolinewidth = 2,
    zerolinecolor = "black",
)

fig.update_layout(legend_orientation="h")
fig.update_layout(
    legend = dict(
                   x = 0.25,
                   y = 1.11,
                   traceorder= "normal",
                   # bordercolor= "Black",
                   # borderwidth= 0.5
    )
)

# fig.update_layout(legend_orientation="h")
# fig.update_layout(
#     legend = dict(
#                    x = 0,
#                    y = 1.1,
#                    traceorder= "normal",
#                    # bordercolor= "Black",
#                    # borderwidth= 0.5
#     )
# )
fig.write_image("analysis/plots/base_analysis/" + SCORE + "_rep_normalized.eps")
fig.write_image("analysis/plots/base_analysis/" + SCORE + "_rep_normalized.png")
fig.write_image("/home/jhosoume/unb/tcc/ICDM/img/base_level_analysis/" + baseline  + "_" + SCORE + "_rep_normalized.eps")
fig.write_image("/home/jhosoume/unb/tcc/ICDM/img/base_level_analysis/" + baseline + "_" + SCORE + "_rep_normalized.png")
