__author__ = 'Aga'

import numpy as np
from collections import Counter
from sklearn.tree import DecisionTreeClassifier

"""
Adaboost for binary classification
"""
class AdaBoost:
    def __init__(self, num_iterations):
        self.num_iterations = num_iterations

    def fit(self, X, y):
        hypotheses = []
        hypothesis_weights = []

        N, _ = X.shape
        d = np.ones(N) / N

        for t in range(self.num_iterations):
            h = DecisionTreeClassifier(max_depth=1)

            h.fit(X, y, sample_weight=d)
            pred = h.predict(X)

            eps = d.dot(pred != y)
            alpha = (np.log(1 - eps) - np.log(eps)) / 2

            d = d * np.exp(- alpha * y * pred)
            d = d / d.sum()

            hypotheses.append(h)
            hypothesis_weights.append(alpha)
        return (hypotheses, hypothesis_weights)


    def predict(self, X, hypotheses, hypotheses_weight):
        y = np.zeros(X.shape)
        for (h, alpha) in zip(hypotheses, hypotheses_weight):
            y = y + alpha * h.predict(X)
        y = np.sign(y)
        b = Counter(list(y))
        return b.most_common(1)[0][0]

