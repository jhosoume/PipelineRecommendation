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

class HARF_AS(AutoSklearnPreprocessingAlgorithm):
    def __init__(self, random_state = None):
        self.random_state = random_state

    def fit(self, X, y):
        self._cur_X = X
        self._cur_y = y
        from NoiseFiltersPy import HARF
        self.preprocessor = HARF(n_jobs = 1, seed = self.random_state)
    
    def transform(self, X):
    # Problem! Chosen pre processors modify data shape :(
        if self.preprocessor is None:
            raise NotImplementedError()
        if X == self._cur_X:
            filter = self.preprocessor(data = X, classes = self._cur_y)
            return filter.cleanData
        return X

    @staticmethod
    def get_properties(dataset_properties = None):
        return {
            'shortname': 'HARF',
            'name': 'High Agreement Random Forest',
            'handles_regression': False,
            'handles_classification': True,
            'handles_multiclass': True,
            'handles_multilabel': False,
            'handles_multioutput': False,
            'is_deterministic': False,
            'input': (DENSE, UNSIGNED_DATA, SIGNED_DATA),
            'output': (DENSE, UNSIGNED_DATA, SIGNED_DATA)
        }

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        cs = ConfigurationSpace()
        return cs 
