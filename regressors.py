import sys
import pandas as pd
import numpy as np

from sklearn import svm, linear_model, discriminant_analysis, neighbors
from sklearn import tree, naive_bayes, ensemble
from sklearn.model_selection import cross_validate
from sklearn.metrics import SCORERS
from sklearn import preprocessing

import constants
from meta_db.db.DBHelper import DBHelper

db = DBHelper()

SCORE_COLUMNS = ["name"] + constants.CLASSIFIERS

def total_score(record):
    result = (record["recall_micro_mean"] + record["recall_macro_mean"] +  # Recall: 1 is the best
              record["accuracy_mean"] +  # Accuracy is normalized
              record["precision_micro_mean"] + record["precision_macro_mean"] +  # Same as recall
              record["f1_micro_mean"] + record["f1_macro_mean"]) # 1 is the best
    return result / 7

data = pd.DataFrame(db.get_all_metadata(), columns = db.metadata_columns())
models = pd.DataFrame(db.get_all_models(), columns = db.models_columns())

scores = pd.DataFrame(columns = SCORE_COLUMNS)
scores["name"] = models["name"].unique()

for indx, record in models.iterrows():
    scores.loc[scores["name"] == record["name"], record["model"]] = total_score(record)

# for indx, score in scores.iterrows():
#     db.add_scores_record(SCORE_COLUMNS, score.values.tolist())

targets = {}
for clf in constants.CLASSIFIERS:
    targets[clf] = pd.DataFrame(db.get_model_record_per_model(clf), columns = db.models_columns())
    targets[clf] = pd.merge(data, targets[clf], on = "name")
