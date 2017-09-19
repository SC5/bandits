# Imports
import numpy as np

from sklearn.exceptions import NotFittedError
from sklearn.linear_model import SGDRegressor, SGDClassifier

class epsilonGreedyContextualBandit(object):

    def __init__(self, epsilon=0.2, fit_intercept=True, penalty='l2'):
        self.config = {
            'epsilon': epsilon,
            'fit_intercept': fit_intercept,
            'penalty': penalty
        }
        self.arms = {}
        self.n_arms = 0

    def select_arm(self, context, arms):
        for arm in arms:
            if arm not in self.arms:
                self.arms[arm] = SGDRegressor(
                    fit_intercept=self.config['fit_intercept'],
                    penalty=self.config['penalty'],
                    max_iter=1,
                    tol=None
                )
                self.n_arms += 1

        if np.random.uniform() <= self.config['epsilon']:
            return np.random.choice(list(self.arms.keys()))
        else:
            try:
                predictions = []
                candidates = []
                arms = self.arms.keys()
                for arm in arms:
                    predictions.append(self.arms[arm].predict(context))
                    candidates.append(arm)
                return candidates[np.argmin(predictions)]
            except NotFittedError:
                return np.random.choice(list(self.arms.keys()))

    def reward(self, arm, context, cost):
        self.arms[arm].partial_fit(context, [cost])
