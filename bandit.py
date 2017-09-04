# Imports
import numpy as np

from sklearn.exceptions import NotFittedError
from sklearn.linear_model import SGDRegressor, SGDClassifier

class unknownModeException(Exception):
    pass

class epsilonGreedyContextualBandit(object):

    def __init__(self, mode='regression', epsilon=0.2, fit_intercept=True, penalty='l2'):
        self.config = {
            'epsilon': epsilon,
            'fit_intercept': fit_intercept,
            'mode': mode,
            'penalty': penalty
        }
        self.arms = {}
        self.n_arms = 0

    def select_arm(self, context, arms):
        for arm in arms:
            if arm not in self.arms:
                # Initialise a new linear regression or classification model for predicting the reward
                if self.config['mode'] is 'regression':
                    self.arms[arm] = SGDRegressor(
                        fit_intercept=self.config['fit_intercept'],
                        penalty=self.config['penalty'],
                        max_iter=1,
                        tol=None
                    )
                elif self.config['mode'] is 'classification':
                    self.arms[arm] = SGDClassifier(
                        fit_intercept=self.config['fit_intercept'],
                        penalty=self.config['penalty'],
                        loss='log',
                        max_iter=1,
                        tol=None
                    )
                else:
                    raise unknownModeException("Unknown mode (must be either 'regression' or 'classification' ")
                self.n_arms += 1

        if np.random.uniform() <= self.config['epsilon']:
            return np.random.choice(list(self.arms.keys()))
        else:
            try:
                predictions = []
                candidates = []
                arms = self.arms.keys()
                for arm in arms:
                    if self.config['mode'] is 'regression':
                        predictions.append(self.arms[arm].predict(context))
                    else:
                        predictions.append(self.arms[arm].predict_proba(context)[0][1])
                    candidates.append(arm)
                return candidates[np.argmax(predictions)]
            except NotFittedError:
                return np.random.choice(list(self.arms.keys()))

    def reward(self, arm, context, reward):
        if self.config['mode'] is 'regression':
            self.arms[arm].partial_fit(context, [reward])
        else:
            self.arms[arm].partial_fit(context, [reward], classes=[0,1])
