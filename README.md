# Contextual bandit example

This repo contains a simple Python implementation of a contextual bandit, and an example showing how to use it to optimise click-though rates for different advertisments. The bandit maintains one regression model per arm, in order to predict the expected cost for each arm (i.e. negative reward). Exploration is done 10% of the time -- you can edit this by changing the `epsilon` parameter in `app.py`.

## Requirements

The bandit requires Python 3.5+ and associated packages `scikit`, `scipy` and `numpy`. To install them all:

     pip install -U scikit-learn scipy numpy

## Running the simulation

    python3 app.py

## Running the demo application

_Note: the demo requires Docker to be installed on your machine_

### Build the Docker image

From the root directory of the repository:

    docker build -t bandit-demo .
    docker run -p 8000:8000 -v $(pwd)/static:/bandit/static bandit-demo

The demo can be run by visiting `http://0.0.0.0:8000` in any browser. If you want to make changes to the demo, edit `static/index.html` and reload the browser tab.

## Todo

- Remove base64 encoding to server logic
- Add support for shared features
- Add support for model loading via queue
- Add support for model training via queue
- Add support for Boltzmann exploration