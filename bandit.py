import numpy as np

from sklearn.exceptions import NotFittedError
from sklearn.linear_model import SGDRegressor
from sklearn.feature_extraction.text import HashingVectorizer


class epsilonGreedyContextualBandit(object):

    def __init__(self, epsilon=0.1, fit_intercept=True, penalty='l2', alpha=0.1, n_features=2**20):
        self.config = {
            'epsilon': epsilon,
            'fit_intercept': fit_intercept,
            'penalty': penalty,
            'alpha': alpha
        }
        self.arms = {}
        self.n_arms = 0
        self.n_features = n_features
        self.vectorizer = HashingVectorizer(self.n_features)

    def select_arm(self, context, choices):
        context = self.vectorizer.fit_transform([context])
        for arm in choices:
            if arm not in self.arms:
                self.arms[arm] = SGDRegressor(
                    fit_intercept=self.config['fit_intercept'],
                    penalty=self.config['penalty'],
                    max_iter=1,
                    alpha=self.config['alpha'],
                    tol=None
                )
                self.n_arms += 1

        if np.random.uniform() <= self.config['epsilon']:
            return (np.random.choice(choices), 'explore')
        else:
            try:
                predictions = []
                candidates = []
                arms = self.arms.keys()
                for arm in arms:
                    predictions.append(self.arms[arm].predict(context))
                    candidates.append(arm)
                return (candidates[np.argmin(predictions)], 'exploit')
            except NotFittedError:
                return (np.random.choice(choices), 'explore')

    def reward(self, arm, context, cost):
        context = self.vectorizer.fit_transform([context])
        self.arms[arm].partial_fit(context, [cost])

    def reset(self):
        self.__init__(
            epsilon=self.config['epsilon'],
            fit_intercept=self.config['fit_intercept'],
            penalty=self.config['penalty'],
            n_features=self.n_features
        )
