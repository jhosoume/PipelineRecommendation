from os import listdir
from os.path import isfile, join
import pandas as pd
import numpy as np
from sklearn import preprocessing
from scipy.io import arff as arff_io
from pymfe.mfe import MFE
from meta_db.db.DBHelper import DBHelper
from config import config

datasets = [f for f in listdir(config["dataset"]["folder"])
                if ( isfile(join(config["dataset"]["folder"], f)) and
                   ( f.endswith("json") or f.endswith("arff") ) )]

db = DBHelper()
mfe = MFE()
le = preprocessing.LabelEncoder()

for dataset in datasets:
    if (dataset[:-5] in db.get_metadata_datasets()):
        print("Dataset already in the database.")
        continue
    if dataset.endswith("json"):
        data = pd.read_json(config["dataset"]["folder"] + dataset)
    elif dataset.endswith("arff"):
        data = arff_io.loadarff(config["dataset"]["folder"] + dataset)
        data = pd.DataFrame(data[0])
    target = data["class"].values
    if target.dtype == np.object:
        le.fit(target)
        target = le.transform(target)
    values = data.drop("class", axis = 1).values
    mfe.fit(values, target)
    try:
        ft = mfe.extract()
    except AttributeError:
        mfe.fit(values.astype(float), target)
        ft = mfe.extract()
    labels = np.array(ft[0])
    results = np.array(ft[1])
    nan_columns = np.isnan(results)
    not_nan = np.invert(nan_columns)
    # Ignoring nan values
    print("Problems with: ", labels[nan_columns])
    # Adding name to the list
    labels = ["name"] + labels[not_nan].tolist()
    results = [dataset[:-5]] + results[not_nan].tolist()
    for indx, result in enumerate(results):
        if isinstance(result, complex):
            results[indx] = result.real
    db.add_metadata_record(labels, results)
