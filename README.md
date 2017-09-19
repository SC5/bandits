# Contextual bandit example
This repo contains a simple Python implementation of a contextual bandit, and an example showing how to use it to optimise click-though rates for different advertisments. The bandit maintains one regression model per arm, in order to predict the expected cost for each arm (i.e. negative reward). Exploration is done 10% of the time -- you can edit this by changing the `epsilon` parameter in `app.py`.

## Requirements
The bandit requires `scikit`, `scipy` and `numpy`. To install them all:

     pip install -U scikit-learn scipy numpy

## Running the simulation

    python3 app.py
