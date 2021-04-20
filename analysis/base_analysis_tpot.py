import os
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn import compose 
from sklearn import metrics
from sklearn.model_selection import train_test_split

from scipy.io import arff as arff_io

from tpot import TPOTClassifier

import constants
from meta_db.db.DBHelper import DBHelper
from config import config



SCORE = "balanced_accuracy_mean"
SCORE_ = "balanced_accuracy"
REPETITIONS = 30

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

tpot_clf = {
    "svc": "svm",
    "logisticregression": "logistic_regression",
    "lineardiscriminantanalysis": "linear_discriminant",
    "kneighborsclassifier": "kneighbors",
    "decisiontreeclassifier": "decision_tree",
    "gaussiannb": "gaussian_nb",
    "randomforestclassifier": "random_forest",
    "gradientboostingclassifier": "gradient_boosting"

}

tpot_conf = {
    "sklearn.svm.SVC": {"gamma": ["auto"]},
    "sklearn.linear_model.LogisticRegression": {"solver": ["lbfgs"]},
    "sklearn.discriminant_analysis.LinearDiscriminantAnalysis": {},
    "sklearn.neighbors.KNeighborsClassifier": {},
    "sklearn.tree.DecisionTreeClassifier": {},
    "sklearn.naive_bayes.GaussianNB": {},
    "sklearn.ensemble.RandomForestClassifier": {"n_estimators": [100]},
    "sklearn.ensemble.GradientBoostingClassifier": {}
}


db = DBHelper()
enc = preprocessing.OneHotEncoder(handle_unknown = 'ignore')
le = preprocessing.LabelEncoder()


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

pipeline_opt = TPOTClassifier(
    generations = 2,
    population_size = 5,
    cv = 10,
    verbosity = 2,
    scoring = SCORE_,
    config_dict = tpot_conf,
    template = "Classiifier",
    memory = "auto"
)

for dataset in datasets.iloc[:1]:
    dataset_info = arff_io.loadarff(config["dataset"]["folder"] + dataset + ".arff")
    dataset_info = pd.DataFrame(dataset_info[0])
    target = dataset_info["class"].values
    # Data preprocessing (type transformation)
    if target.dtype == object:
        le.fit(target)
        target = le.transform(target)
    attrs = dataset_info.drop("class", axis = 1)
    if np.any(attrs.dtypes == object):
        ct = compose.ColumnTransformer(
            transformers = [("encoder", enc, attrs.dtypes == object)],
            remainder = "passthrough"
            )
        attrs = ct.fit_transform(attrs)
    attrs = np.array(attrs)
    X_train, X_test, y_train, y_test = train_test_split(attrs, target, test_size = 0.2)
    pipeline_opt.fit(X_train, y_train)
    recommended_clf = pipeline_opt.fitted_pipeline_.steps[0][0]
    import ipdb; ipdb.set_trace()