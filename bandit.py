# Imports
import random
import numpy as np

from sklearn.exceptions import NotFittedError
from sklearn.linear_model import SGDRegressor

class epsilonGreedyContextualBandit(object):

    def __init__(self, epsilon=0.2, fit_intercept=True):
        self.config = {
            'epsilon': epsilon,
            'fit_intercept': fit_intercept
        }
        self.arms = {}
        self.n_arms = 0

    def select_arm(self, context, arms):
        for arm in arms:
            if arm not in self.arms:
                # Initialise a new linear regression model for predicting the reward
                self.arms[arm] = SGDRegressor(
                    fit_intercept=self.config['fit_intercept'],
                    max_iter=5,
                    tol=None
                )
                self.n_arms += 1
        if random.random() <= self.config['epsilon']:
            return random.choice(list(self.arms.keys()))
        else:
            try:
                predictions = []
                candidates = []
                arms = self.arms.keys()
                for arm in self.arms.keys():
                    predictions.append(self.arms[arm].predict(context))
                    candidates.append(arm)
                return candidates[np.argmax(predictions)]
            except NotFittedError:
                return random.choice(list(self.arms.keys()))

    def reward(self, arm, context, reward):
        self.arms[arm].partial_fit(context, [reward])
