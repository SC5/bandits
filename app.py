import sys
import numpy as np
import bandit

# np.random.seed(2)

contextual_bandit = bandit.epsilonGreedyContextualBandit(epsilon=0.1, penalty='l1')

# Example with advertisments we would like to show
arms = ['advertisement_1',
        'advertisement_2',
        'advertisement_3',
        'advertisement_4',
        'advertisement_5',
        'advertisement_6',
        'advertisement_7',
        'advertisement_8']

# The job of the bandit is to learn the true click-through rates
# of each arm, but for simulation purposes, we'll cheat and pretend
# we already know.
ctrs = [0.993, 0.0521, 0.0122, 0.05215, 0.074, 0.0521, 0.07482, 0.0154]

# Simulate a single context, a male website visitor aged 21 that uses Firefox
context = 'age_21 gender_male browser_firefox'

counts = np.zeros(len(arms)) # Keep count of how many times each arm was chosen
costs = np.zeros(len(arms)) # Keep count of the costs for each arm

epochs = 10000
sys.stdout.write('Running simulation for ' + str(epochs) + ' epochs')
for i in range(epochs):
    sys.stdout.write('.')
    sys.stdout.flush()
    chosen_arm, phase, predictions = contextual_bandit.select_arm(context, arms)
    # Send cost based on our pretend CTR for the chosen arm:
    # - -1: clicked
    # - 1: not clicked
    if np.random.random() < ctrs[arms.index(chosen_arm)]:
        contextual_bandit.reward(chosen_arm, context, 0)
        costs[arms.index(chosen_arm)] += 0
    else:
        contextual_bandit.reward(chosen_arm, context, 1)
        costs[arms.index(chosen_arm)] += 1
    counts[arms.index(chosen_arm)] += 1

print('done.\nResults:')
for i, v in enumerate(counts):
    print('Arm ' + arms[i] + ' was chosen ' + str(counts[i]) + ' times, with a cumulative cost of ' + str(costs[i]) + ' (cost per play: ' + str(costs[i] / counts[i]) + ')')
