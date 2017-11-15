import base64
import json
import numpy as np

from sklearn.exceptions import NotFittedError
from sklearn.linear_model import SGDRegressor
from sklearn.feature_extraction.text import HashingVectorizer

class epsilonGreedyContextualBandit(object):

    def __init__(
            self,
            epsilon=0.1,
            fit_intercept=True,
            penalty='l2',
            ips=True,
            learning_rate=0.01,
            n_features=1024,
            mode='online',
            batch_size=128,
            burn_in=1000
            ):
        self.config = {
            'epsilon': epsilon,
            'fit_intercept': fit_intercept,
            'penalty': penalty,
            'learning_rate': learning_rate,
            'mode': mode,
            'batch_size': batch_size,
            'ips': ips,
            'burn_in': burn_in
        }
        self.arms = {}
        self.n_arms = 0
        self.n_features = n_features
        self.vectorizer = HashingVectorizer(self.n_features)
        self.batch = []
        self.batch_counter = 0
        self.epoch = 0
        self.model = SGDRegressor(
            fit_intercept=self.config['fit_intercept'],
            penalty=self.config['penalty'],
            max_iter=1,
            eta0=self.config['learning_rate'],
            learning_rate='constant',
            tol=None
        )

    def _explode_features(self, context, choice, return_array=True):
        prefixed_words = [choice + '_' + w for w in context.split(' ')]
        context = ' '.join(prefixed_words)
        if return_array:
            return [context]
        else:
            return context

    def _explode_features_batch(self, context, choices):
        exploded_contexts = []
        for c in choices:
            prefixed_words = [c + '_' + w for w in context.split(' ')]
            exploded_features = ' '.join(prefixed_words)
            exploded_contexts.append(exploded_features)
        return exploded_contexts

    def _weight(self, reward, prob):
        if self.config['ips']:
            return self._ips_weight(reward, prob)
        else:
            return -reward

    def _ips_weight(self, reward, prob):
        return (-reward) / prob

    def _prob_dist(self, n, opt_idx, randomise=False):
        epsilon = self.config['epsilon']
        dist = np.full(n, epsilon / n)
        if randomise:
            opt_idx = numpy.random.randint(0, n)
        dist[opt_idx] = (1 - epsilon) + (epsilon / n)
        return dist

    def _get_prob(self, n, choice, opt_choice):
        epsilon = self.config['epsilon']
        if choice == opt_choice:
            return (1 - epsilon) + (epsilon / n)
        else:
            return epsilon / n

    def select_arm(self, context, choices):
        self.epoch += 1
        contexts = self._explode_features_batch(context, choices)
        contexts = self.vectorizer.fit_transform(contexts)

        try:
            predictions = self.model.predict(contexts)
            opt_idx = np.argmin(predictions)
        except NotFittedError:
            predictions = []
            opt_idx = 0

        choice = np.random.choice(choices, p=self._prob_dist(len(choices), opt_idx))

        decision = base64.b64encode(json.dumps({
            'choices': choices,
            'choice': choice,
            'prob': self._get_prob(len(choices), choice, choices[opt_idx])
        }).encode())

        return (choice, predictions, decision)

    def reward(self, context, reward, decision):
        decision = json.loads(base64.b64decode(decision))
        choice = decision['choice']
        choices = decision['choices']
        choices.remove(choice)
        cost = self._weight(reward, decision['prob'])

        if self.config['mode'] == 'online':
            exploded_context = self._explode_features(context, choice)
            self.model.partial_fit(self.vectorizer.fit_transform(exploded_context), [cost])
            if self.config['ips']:
                exploded_contexts = self._explode_features_batch(context, choices)
                self.model.partial_fit(self.vectorizer.fit_transform(exploded_contexts), np.full(len(choices), 0))

        else:
            exploded_context = self._explode_features(context, choice, return_array=False)
            self.batch.append((exploded_context, cost))
            if self.config['ips']:
                exploded_contexts = self._explode_features_batch(context, choices)
                for item in exploded_contexts:
                    self.batch.append((item, 0))

            self.batch_counter += 1
            if self.batch_counter == self.config['batch_size']:
                self._batch_reward(self.batch)
                self.batch = []
                self.batch_counter = 0

    def _batch_reward(self, batch):
        contexts, costs = map(list, zip(*batch))
        self.model.partial_fit(self.vectorizer.fit_transform(contexts), costs)

    def reset(self):
        self.__init__(
            epsilon=self.config['epsilon'],
            fit_intercept=self.config['fit_intercept'],
            penalty=self.config['penalty'],
            learning_rate=self.config['learning_rate'],
            n_features=self.n_features,
            mode=self.config['mode'],
            batch_size=self.config['batch_size'],
            ips=self.config['ips'],
            burn_in=self.config['burn_in']
        )
