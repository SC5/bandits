import sys
import numpy as np
import bandit

np.random.seed(12345)

contextual_bandit = bandit.epsilonGreedyContextualBandit(epsilon=0.1, penalty='None', mode='batch', batch_size=128, ips=True)
print('Bandit configuration', contextual_bandit.config)

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
rewards = np.zeros(len(arms)) # Keep count of the rewards for each arm

epochs = 50000
sys.stdout.write('Running simulation for ' + str(epochs) + ' epochs')
for i in range(epochs):
    sys.stdout.write('.')
    sys.stdout.flush()
    chosen_arm, predictions, decision_id = contextual_bandit.select_arm(context, arms) 
    # Send reward based on our pretend CTR for the chosen arm:
    # - 1: clicked
    # - 0: not clicked
    if np.random.random() < ctrs[arms.index(chosen_arm)]:
        contextual_bandit.reward(context, 1, decision_id)
        rewards[arms.index(chosen_arm)] += 1
    else:
        contextual_bandit.reward(context, 0, decision_id)
        rewards[arms.index(chosen_arm)] += 0
    counts[arms.index(chosen_arm)] += 1


print('done.\nResults:')
for i, v in enumerate(counts):
    print('Arm ' + arms[i] + ' was chosen ' + str(counts[i]) + ' times, with a cumulative reward of ' + str(rewards[i]) + ' (avg. reward per play: ' + str(rewards[i] / counts[i]) + ')')
