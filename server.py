import bandit
import logging

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from sanic import Sanic
from sanic.response import json

app = Sanic()
bandits = {}
bandits['default'] = bandit.epsilonGreedyContextualBandit(alpha=0.5, penalty='l1', epsilon=0.2)

app.static('', './static/index.html')
app.static('/', './static')

logger = logging.getLogger('sanic')

# async def tick():
#     logger.info('Tick! The time is: %s' % datetime.now())
#     time.sleep(1)

# @app.listener('before_server_start')
# async def instantiate_scheduler(app, loop):
#     scheduler = AsyncIOScheduler({'event_loop': loop})
#     scheduler.add_job(tick, 'interval', seconds=1)
#     scheduler.start()

@app.route("/predict/<id>", methods=["POST"])
async def predict(request, id):
    if id not in bandits:
        bandits[id] = bandit.epsilonGreedyContextualBandit()
    body = request.json
    context = body['context']
    arms = body['arms']
    arm, phase, predictions= bandits[id].select_arm(context, arms)
    return json({"arm": arm, "phase": phase, "predictions": predictions})

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