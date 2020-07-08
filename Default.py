import random
import numpy as np

class Default:
    def __init__(self):
        self.target = []

    def fit(self, values, target):
        self.target = target
        self.mean = np.mean(target)
        return self

    def predict(self, values):
        return np.array([self.mean for num in range(len(values))])
