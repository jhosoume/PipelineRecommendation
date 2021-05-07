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

class ENNS(AutoSklearnPreprocessingAlgorithm):
    def __init__(self, random_state = None):
        self.random_state = random_state
        self.preprocessor = None

    def fit(self, X, y = None):
        print('Im here')
        self._cur_X = X
        self._cur_y = y
        from NoiseFiltersPy import ENN
        self.preprocessor = ENN(n_jobs = 1)
        return self
    
    def transform(self, X):
        print(X == self._cur_X)
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
            'shortname': 'ENNS',
            'name': 'ENN',
            'handles_regression': False,
            'handles_classification': True,
            'handles_multiclass': True,
            'handles_multilabel': False,
            'handles_multioutput': False,
            'is_deterministic': True,
            'input': (DENSE, UNSIGNED_DATA, SIGNED_DATA),
            'output': (DENSE, UNSIGNED_DATA, SIGNED_DATA)
        }

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        cs = ConfigurationSpace()
        return cs 