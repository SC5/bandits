import sys
import numpy as np
import bandit

np.random.seed(1337)

from sklearn.feature_extraction.text import HashingVectorizer

# Use the hashing trick to make feature vectors of uniform length (n_features) regardless
# of the number of features passed in the context (remaning features are set to zero)
vectorizer = HashingVectorizer(n_features=1024)
contextual_bandit = bandit.epsilonGreedyContextualBandit(mode='classification', epsilon=0.2, penalty='l2')

# Example with three advertisments we would like to show
arms = ['advertisment_1', 
        'advertisment_2', 
        'advertisment_3', 
        'advertisment_4', 
        'advertisment_5', 
        'advertisment_6', 
        'advertisment_7', 
        'advertisment_8']

# The job of the bandit is to learn the true click-through rates
# of each arm, but for simulation purposes, we'll cheat and pretend
# we already know.
ctrs = [0.076, 0.0521, 0.0122, 0.05215, 0.074, 0.0521, 0.07582, 0.0154]

# Simulate a single context, a male website visitor aged 21 that uses Firefox
context = vectorizer.fit_transform(['age_21 gender_male browser_firefox'])

counts = np.zeros(len(arms)) # Keep count of how many times each arm was chosen
rewards = np.zeros(len(arms)) # Keep count of the rewards for each arm

epochs = 10000
sys.stdout.write('Running simulation for ' + str(epochs) + ' epochs')
for i in range(epochs):
    sys.stdout.write('.')
    sys.stdout.flush()
    chosen_arm = contextual_bandit.select_arm(context, arms)
    # Send reward based on our pretend CTR for the chosen arm:
    # - 1: clicked
    # - 0: not clicked
    if np.random.random() <= ctrs[arms.index(chosen_arm)]:
        contextual_bandit.reward(chosen_arm, context, 1)
        rewards[arms.index(chosen_arm)] += 1
    else:
        contextual_bandit.reward(chosen_arm, context, 0)
        rewards[arms.index(chosen_arm)] += 0
    counts[arms.index(chosen_arm)] += 1

print('done.\nResults:')
for i, v in enumerate(counts):
    print('Arm ' + arms[i] + ' was chosen ' + str(counts[i]) + ' times, with a cumulative reward of ' + str(rewards[i]) + '.')
