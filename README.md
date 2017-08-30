# Contextual bandit example
This repo contains a simple Python implementation of a contextual bandit. The bandit maintains one linear regression model per arm, in order to predict the expected reward. Exploration is done 10% of the time.


## Requirements
The bandit requires `scikit` and `numpy`. To install both:

     pip install -U scikit-learn numpy

## Running the simulation

    python3 app.py