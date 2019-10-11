import os
import pandas as pd
import openml
import arff

PATH = "datasets"

# Saving selected dataset from openml to PATH in JSON format
# Format chosen: https://towardsdatascience.com/the-best-format-to-save-pandas-data-414dca023e0d
# https://medium.com/@shmulikamar/python-serialization-benchmarks-8e5bb700530b
def save_dataset(data_id):
    try:
        dataset = openml.datasets.get_dataset(int(data_id))
    except:
        # Some databases in openml has problemas with '' in ARFF, skiping those
        print("ERROR! {}".format(data_id))
        return
    # Need to check if it is a classification problem
    # Verify if the column with the label is called class
    data, target_column, categorical_indicator, attribute_names = dataset.get_data(
        dataset_format = 'dataframe',
        target = dataset.default_target_attribute
    )
    data['class'] = target_column
    print("Saving dataset {} to Json.\n".format(dataset.name))
    # Save as ARFF, needs to be a binary
    data.to_json("datasets/{}.json".format(dataset.name))
    return

# By default, active datasets are returned
datalist = openml.datasets.list_datasets(output_format = 'dataframe',
                                         number_missing_values = 0)

datalist = datalist.query("NumberOfInstances <= 15000 and \
                           NumberOfInstances > 100 and \
                           NumberOfFeatures <= 20 and \
                           NumberOfFeatures > 4 and \
                           NumberOfClasses >= 2 and \
                           NumberOfClasses <= 10 and \
                           MinorityClassSize > 10 \
                          ")
try:
    os.mkdir(PATH)
except FileExistsError:
    print("Directory already exists. Datasets will be appended.")
# Saving each found database
for index, dataset in datalist.iterrows():
    print(index)
    save_dataset(dataset.did)
