from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, CategoricalHyperparameter
from ConfigSpace.conditions import InCondition

import autosklearn.pipeline.components.feature_preprocessing
from autosklearn.pipeline.components.base \
    import AutoSklearnPreprocessingAlgorithm
from autosklearn.pipeline.constants import DENSE, SIGNED_DATA, \
    UNSIGNED_DATA
from autosklearn.util.common import check_none

class HARFS(AutoSklearnPreprocessingAlgorithm):
    def __init__(self, random_state = None):
        self.random_state = random_state
        self.preprocessor = None

    def fit(self, X, y = None):
        self._cur_X = X
        self._cur_y = y
        from NoiseFiltersPy.HARF import HARF
        self.preprocessor = HARF(n_jobs = 1, seed = self.random_state)
        return self
    
    def transform(self, X):
    # Problem! Chosen pre processors modify data shape :(
        if self.preprocessor is None:
            raise NotImplementedError()
        print(self._cur_y is not None)
        print(X == self._cur_X)
        #if X == self._cur_X and (self._cur_y is not None):
            #2 + 2
        print(self.preprocessor)
        print(len(X), len(self._cur_y))
        #filter = self.preprocessor(data = X, classes = self._cur_y)
        #print(filter.cleanData)
        # return filter.cleanData
        return X

    @staticmethod
    def get_properties(dataset_properties = None):
        return {
            'shortname': 'HARFS',
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