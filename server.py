import bandit
import logging

from sanic import Sanic
from sanic.response import json

app = Sanic()
contextual_bandit = bandit.epsilonGreedyContextualBandit(epsilon=0.1, penalty='l2')

app.static('/', './static')
logger = logging.getLogger('sanic')

@app.route("/predict", methods=["POST"])
async def predict(request):
    body = request.json
    context = body['context']
    arms = body['arms']
    arm = contextual_bandit.select_arm(context, arms)
    return json({"arm": arm})

@app.route("/reward", methods=["POST"])
async def reward(request):
    body = request.json
    context = body['context']
    arm = body['arm']
    cost = body['cost']
    contextual_bandit.reward(arm, context, cost)
    return json({"cost": cost})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)