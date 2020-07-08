import random
import constants
import numpy as np


class Random:
    def __init__(self, random_seed = constants.RANDOM_STATE):
        random.seed(random_seed)
        self.target = []

    def fit(self, values, target):
        self.target = target
        return self

    def predict(self, values):
        return np.array([random.choice(self.target) for i in range(len(values))])
