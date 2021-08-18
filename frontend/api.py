import json
import logging
from datetime import datetime

import flask
from flask_cors import CORS
import click
from gevent.pywsgi import WSGIServer

from best_of_n.generator import BestOfNGenerator

SAMPLING_KEEP_TOP_P = 0.95
BEST_OF_N_N = 100
REWARD_MODEL_CKPT_PATH = "gs://"
T5_MODEL_CKPT_STEPS = 1010000
T5_MODEL_CKPT_PATH = f"gs://seri2021-advice/turingadvice/baselines/t5/11B/model.ckpt-{T5_MODEL_CKPT_STEPS}"
TEMPLATE_DIR = "./frontend"

# Initialize models and Best-of-N generator
reward_model = None
t5_model = None
BoN_generator = BestOfNGenerator(
    t5_model=t5_model,
    t5_model_ckpt_steps=T5_MODEL_CKPT_STEPS,
    N=BEST_OF_N_N,
    sampling_keep_top_p=SAMPLING_KEEP_TOP_P
)

# Initialize API
app = flask.Flask(__name__, template_folder=TEMPLATE_DIR)
CORS(app, resources={r'/api/*': {'origins': '*'}})
logger = logging.getLogger(__name__)

def _datetime_to_str(date):
    return [
        'January', 'February', 'March', 'April', 'May', 'June', 'July',
        'August', 'September', 'October', 'November', 'December'
        ][date.month - 1] + ' {}, {}'.format(date.day, date.year)

@app.route('/api/askbatch', methods=['POST'])
def api_askbatch():
    request_dict = dict(flask.request.json)
    instances = request_dict["instances"]
    date = datetime.utcnow()
    date_str = _datetime_to_str(date)
    for instance in instances:
        instance["date"] = date_str
    advices = BoN_generator.predict_from_instances(instances)
    request_dict.update({"advices": advices})
    with open("./frontend/log.jsonl", "a+") as logfile:
        logfile.write(json.dumps(request_dict) + "\n")
    return flask.jsonify({"gens": advices}), 200

@click.command()
def serve():
    """Serve predictions on port 5000."""
    logging.basicConfig(
        format='%(asctime)s:%(levelname)s:%(name)s:%(message)s',
        level=logging.INFO)
    logger.info('Running prod server on http://127.0.0.1:5000/')

WSGIServer(('0.0.0.0', 5000), app).serve_forever()
