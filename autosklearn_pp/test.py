from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, CategoricalHyperparameter
from ConfigSpace.conditions import InCondition

import sklearn.metrics
import autosklearn.classification
import autosklearn.pipeline.components.feature_preprocessing
from autosklearn.pipeline.components.base \
    import AutoSklearnPreprocessingAlgorithm
from autosklearn.pipeline.constants import DENSE, SIGNED_DATA, \
    UNSIGNED_DATA
from autosklearn.util.common import check_none

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from autosklearn_pp.lda_autosklearn import LDA
from autosklearn_pp.harf_autosklearn import HARFS
from autosklearn_pp.enn_autosklearn import ENNS


# Add LDA component to auto-sklearn.
autosklearn.pipeline.components.feature_preprocessing.add_preprocessor(LDA)
autosklearn.pipeline.components.feature_preprocessing.add_preprocessor(ENNS)
autosklearn.pipeline.components.feature_preprocessing.add_preprocessor(HARFS)

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y)

clf = autosklearn.classification.AutoSklearnClassifier(
    time_left_for_this_task=30,
    include_preprocessors=['HARFS'],
    # Bellow two flags are provided to speed up calculations
    # Not recommended for a real implementation
    initial_configurations_via_metalearning=0,
    smac_scenario_args={'runcount_limit': 5},
)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("accuracy: ", sklearn.metrics.accuracy_score(y_pred, y_test))
print(clf.show_models())