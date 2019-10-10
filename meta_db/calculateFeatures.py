from os import listdir
from os.path import isfile, join
import pandas as pd
import numpy as np
from pymfe.mfe import MFE
from meta_db.db.DBHelper import DBHelper

datasets = [f for f in listdir("datasets/")
                if ( isfile(join("datasets/", f)) and f.endswith("json"))]

db = DBHelper()
mfe = MFE()
for dataset in datasets:
    data = pd.read_json("datasets/" + dataset)
    target = data["class"].values
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
    db.add_metadata_record(labels, results)
