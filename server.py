import bandit
import logging

from sanic import Sanic
from sanic.response import json

app = Sanic()
bandits = {}
bandits['default'] = bandit.epsilonGreedyContextualBandit()

app.static('/', './static')
logger = logging.getLogger('sanic')

@app.route("/predict/<id>", methods=["POST"])
async def predict(request, id):
    if id not in bandits:
        bandits[id] = bandit.epsilonGreedyContextualBandit()
    body = request.json
    context = body['context']
    arms = body['arms']
    arm = bandits[id].select_arm(context, arms)
    return json({"arm": arm})

@app.route("/reward/<id>", methods=["POST"])
async def reward(request, id):
    body = request.json
    context = body['context']
    arm = body['arm']
    cost = body['cost']
    bandits[id].reward(arm, context, cost)
    return json({"cost": cost})

@app.route("/reset/<id>", methods=["POST"])
async def reset(request, id):
    bandits[id].reset()
    return json({"reset": "ok"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)