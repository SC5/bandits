import bandit
import logging

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from sanic import Sanic
from sanic.response import json

app = Sanic()
bandits = {}
bandits['default'] = bandit.epsilonGreedyContextualBandit(learning_rate=0.2, penalty='None', epsilon=0.2, mode='online', batch_size=10, fit_intercept=False)

app.static('', './static/index.html')
app.static('/', './static')

logger = logging.getLogger('sanic')
logger.info(bandits['default'].config)

# async def tick():
#     logger.info('Tick! The time is: %s' % datetime.now())
#     time.sleep(1)

# @app.listener('before_server_start')
# async def instantiate_scheduler(app, loop):
#     scheduler = AsyncIOScheduler({'event_loop': loop})
#     scheduler.add_job(tick, 'interval', seconds=1)
#     scheduler.start()

@app.route("/health")
async def ping(request):
    return json({
        "status": "ok"
    })

@app.route("/predict/<id>", methods=["POST"])
async def predict(request, id):
    if id not in bandits:
        bandits[id] = bandit.epsilonGreedyContextualBandit()
    body = request.json
    context = body['context']
    arms = body['arms']
    arm, predictions, decision_id = bandits[id].select_arm(context, arms)
    return json({
        "arm": arm,
        "predictions": predictions,
        "decision_id": decision_id
    })

@app.route("/reward/<id>", methods=["POST"])
async def reward(request, id):
    body = request.json
    context = body['context']
    decision_id = body['decision_id']
    reward = body['reward']
    bandits[id].reward(context, reward, decision_id)
    return json({"reward": reward})

@app.route("/reset/<id>", methods=["POST"])
async def reset(request, id):
    bandits[id].reset()
    return json({"reset": "ok"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
