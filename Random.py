import random

class Random:
    def __init__(self):
        self.target = []

    def fit(self, values, target):
        self.target = target
        return self

    def predict(self, values):
        return [random.choice(self.target) for i in range(len(values))]
