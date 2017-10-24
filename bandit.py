import base64
import json
import numpy as np

from sklearn.exceptions import NotFittedError
from sklearn.linear_model import SGDRegressor
from sklearn.feature_extraction.text import HashingVectorizer

class epsilonGreedyContextualBandit(object):

    def __init__(self, epsilon=0.1, fit_intercept=True, penalty='l2', alpha=0.01, n_features=32, mode='online', batch_size=128, burn_in=1):
        self.config = {
            'epsilon': epsilon,
            'fit_intercept': fit_intercept,
            'penalty': penalty,
            'alpha': alpha,
            'mode': mode,
            'batch_size': batch_size,
            'burn_in': burn_in
        }
        self.arms = {}
        self.n_arms = 0
        self.n_features = n_features
        self.vectorizer = HashingVectorizer(self.n_features)
        self.batch = []
        self.batch_counter = 0
        self.epoch = 0

    def _ips_weight(self, reward, prob):
        return (-reward) / prob

    def select_arm(self, context, choices):
        self.epoch += 1
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

        if np.random.uniform() <= self.config['epsilon'] or self.epoch <= self.config['burn_in']:
            choice = np.random.choice(choices)
            prob = self.config['epsilon'] / len(choices)
            decision_id = base64.b64encode(json.dumps({
                'choices': choices,
                'choice': choice,
                'prob': prob
            }).encode())
            return (choice, 'explore', [], decision_id)
        else:
            try:
                predictions = []
                candidates = []
                arms = self.arms.keys()
                for arm in arms:
                    predictions.append(self.arms[arm].predict(context))
                    candidates.append(arm)
                choice = candidates[np.argmin(predictions)]
                prob = (1 - self.config['epsilon']) + (self.config['epsilon'] / len(choices))
                decision_id = base64.b64encode(json.dumps({
                    'choices': choices,
                    'choice': choice,
                    'prob': prob
                }).encode())
                return (choice, 'exploit', predictions, decision_id)
            except NotFittedError:
                choice = np.random.choice(choices)
                prob = self.config['epsilon'] / len(choices)
                decision_id = base64.b64encode(json.dumps({
                    'choices': choices,
                    'choice': choice,
                    'prob': prob
                }).encode())
                return (np.random.choice(choices), 'explore', [], decision_id)

    def reward(self, context, reward, decision_id):
        if self.config['mode'] == 'online':
            decision_id = json.loads(base64.b64decode(decision_id))
            arm_played = decision_id['choice']
            arms = decision_id['choices']
            weighted_cost = self._ips_weight(reward, decision_id['prob'])
            context = self.vectorizer.fit_transform([context])
            for arm in arms:
                if arm != arm_played:
                    self.arms[arm].partial_fit(context, [0])
                else:
                    self.arms[arm].partial_fit(context, [weighted_cost])
        else:
            self.batch.append((context, reward, decision_id))
            self.batch_counter += 1
            if self.batch_counter == self.config['batch_size']:
                self._batch_reward(self.batch)
                self.batch = []
                self.batch_counter = 0

    def _batch_reward(self, batch):
        arms_to_fit = {}
        for item in batch:
            context, reward, decision_id = item
            decision_id = json.loads(base64.b64decode(decision_id))
            arm_played = decision_id['choice']
            arms = decision_id['choices']
            weighted_cost = self._ips_weight(reward, decision_id['prob'])
            for arm in arms:
                if arm not in arms_to_fit:
                    arms_to_fit[arm] = {
                        'contexts': [context],
                        'rewards': []
                    }
                else:
                    arms_to_fit[arm]['contexts'].append(context)
                if arm != arm_played:
                    arms_to_fit[arm]['rewards'].append(0)
                else:
                    arms_to_fit[arm]['rewards'].append(weighted_cost)
        for arm in arms_to_fit:
            contexts = self.vectorizer.fit_transform(arms_to_fit[arm]['contexts'])
            self.arms[arm].partial_fit(contexts, arms_to_fit[arm]['rewards'])

    def reset(self):
        self.__init__(
            epsilon=self.config['epsilon'],
            fit_intercept=self.config['fit_intercept'],
            penalty=self.config['penalty'],
            alpha=self.config['alpha'],
            n_features=self.n_features,
            mode=self.config['mode'],
            batch_size=self.config['batch_size'],
            burn_in=self.config['burn_in']
        )
