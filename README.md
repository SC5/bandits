# Contextual bandit example
This repo contains a simple Python implementation of a contextual bandit. The bandit maintains one classification model per arm, in order to predict the expected reward for advertisment optimisation problem (i.e. click-through rate). Exploration is done 10% of the time -- you can edit this by changing the `epsilon` parameter in `app.py`.

For regression problems, where the reward is not binary but a real number, change `mode='classification'` to `mode='regression'`


## Requirements
The bandit requires `scikit` and `numpy`. To install both:

     pip install -U scikit-learn numpy

## Running the simulation

    python3 app.py