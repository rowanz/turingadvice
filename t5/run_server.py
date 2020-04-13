import json
import os
import sys

import flask
import tensorflow as tf
from flask_cors import CORS
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-size', type=str, default="small")
args = parser.parse_args()


sys.path.insert(0, '../')
import logging
from datetime import datetime
import click
from gevent.pywsgi import WSGIServer
from data.to_tfrecord_t5 import _fix_reddit_text, _trim_to_desired_length, encoder
import t5
import re

app = flask.Flask(__name__, template_folder='.')
CORS(app, resources={r'/api/*': {'origins': '*'}})

logger = logging.getLogger(__name__)

# SETUP
top_p = 0.94

model_types = {
    '11B': ('gs://replace-this-with-your-copied-checkpoint-path-please-11B/model.ckpt-1010000', 8, 1),
    '3B': ('gs://replace-this-with-your-copied-checkpoint-path-please-3B/model.ckpt-1018748', 8, 1),
    'large': (
        'gs://replace-this-with-your-copied-checkpoint-path-please-large/model.ckpt-1038196', 8, 1),
    'base': (
        'gs://replace-this-with-your-copied-checkpoint-path-please-base/model.ckpt-1046772', 8, 1),
    'small': (
        'gs://replace-this-with-your-copied-checkpoint-path-please-small/model.ckpt-1037496', 1, 8)
}

def load_estimator_and_predict_items(items, date, model_size):
    """
    :param items: dicts
    :param model_size: model to use
    :return:
    """
    # Choose lowest ppl
    ckpt_path, model_parallelism, batch_size = model_types[model_size]

    print("SIZE={}, ckpt_path={}, model_parallelism={}".format(model_size, ckpt_path, model_parallelism), flush=True)
    ckpt_steps = int(ckpt_path.split('-')[-1])

    model = t5.models.MtfModel(
        model_dir=os.path.dirname(ckpt_path),
        tpu=os.uname()[1],
        tpu_topology='2x2',  # Must be this for validation
        model_parallelism=model_parallelism,
        batch_size=batch_size,
        sequence_length={"inputs": 1280, "targets": 512},
    )

    tmp_input_path = os.path.join(os.path.dirname(ckpt_path), 'tmp_input.txt')
    tmp_output_path = os.path.join(os.path.dirname(ckpt_path), 'tmp_output.txt')

    ctxs = []
    for item in items:
        date_txt = ['January', 'February', 'March', 'April', 'May', 'June', 'July',
                    'August', 'September', 'October', 'November', 'December'][date.month - 1] + ' {}, {}'.format(
            date.day, date.year)

        article_pieces = {}
        article_pieces['subreddit'] = item['subreddit']
        article_pieces['date'] = date_txt
        article_pieces['title'] = item['title']
        article_pieces['selftext'] = _trim_to_desired_length(encoder, item['selftext'], desired_len=1250)
        ex = {k: _fix_reddit_text(v) for k, v in article_pieces.items()}
        ctx = ''.join(
            ["Subreddit: ", ex['subreddit'], " Date: ", ex['date'], " Title: ", ex['title'], " Selftext: ", ex['selftext']])
        ctxs.append(ctx)

    for _ in range(len(items) % 8):
        ctxs.append(ctxs[-1])

    with tf.io.gfile.GFile(tmp_input_path, 'w') as f:
        for ctx in ctxs:
            f.write(ctx + '\n')

    model.predict(
        input_file=tmp_input_path,
        output_file=tmp_output_path,
        checkpoint_steps=ckpt_steps,
        sampling_keep_top_p=top_p
    )
    with tf.io.gfile.GFile(tmp_output_path + f'-{ckpt_steps}', 'r') as f:
        texts = [re.sub(r'\s+»\s+', '\n\n', text).strip() for text in f.read().splitlines()][:len(items)]
    return texts

def load_estimator_and_predict_item(subreddit, title, selftext, model_size):
    return load_estimator_and_predict_items([{'subreddit': subreddit, 'title': title, 'selftext': selftext}], date=datetime.utcnow(), model_size=model_size)[0]


# # Problem: this requires model_parallelism = 1.
# for i in range(3):
#     start = time.time()
#     res = predict_item('relationships', date=datetime.utcnow(), title='My [20M] girlfriend [18F] of 3 months mom caught us talking on facetime while she was in the shower. How is this gonna affect the relationship?',
#                  selftext="Me and my girlfriend have been dating for 3 months and every thing is going great. We were talking on facetime and her mom was supposed to be away for the weekend, so things were getting a little freaky on the call. She then got in the shower and had the camera pointed at her, so that i could see her clearly in the shower while we talked. All of a sudden her mom came in and i heard her yelling so i paused myself but her camera was still on. When i wasnt so shocked i went back to hang up the call and i saw that her mom moved her phone to the sink. I texted my girlfriend after to see if she was ok and this was her response “She just took my phone and put it in the sink and she was like that guy doesn’t let u breath u guys are always talking I don’t understand why u guys ft while I shower u do not respect ur self or make him respect u” and that “And then she was like I will make ur dad speak with him”. I’ve met both her parents once before, and made sure that i left a good impression on them but now this happened. My girlfriend is from a colombian background so i don’t know how they will react to this! How is this going to affect the relationship going into the future?\n\nTLDR: Girlfriend’s mom caught us facetiming while my girlfriend was in the shower and the mom wants the dad to speak with me.")
#     print("Elapsed {:.1f}sec".format(time.time()-start))

@app.route('/', methods=['GET'])
def form_ask():
    """Return the demo page."""
    return flask.render_template('index.html',
                                 model_details="This is T5, default size is 11B but I don't think it's very good TBH. Generation will take 3 minutes.")


@app.route('/api/ask', methods=['POST'])
def api_ask():
    """Serve a prediction for a single instance."""
    instance = dict(flask.request.json)
    print("GOT A REQUEST for {}".format(instance), flush=True)

    if instance['target'] != 'advice':
        return flask.jsonify({
            'instance': instance,
            'gen': instance[instance['target']],
        }), 200

    if 'subreddit' not in instance:
        instance['subreddit'] = 'Advice'

    instance['model_size'] = instance.get('model_size', args.size)
    instance['advice'] = load_estimator_and_predict_item(instance['subreddit'],
                                                         instance.get('title', ''),
                                                         instance.get('selftext', ''),
                                                         model_size=instance['model_size'])
    with open(f'log.jsonl', 'a+') as logfile:
        logfile.write(json.dumps(instance) + '\n')

    return flask.jsonify({
        'instance': instance,
        'gen': instance['advice'],
    }), 200

@app.route('/api/askbatch', methods=['POST'])
def api_askbatch():
    """
    instance has fields instances, each with subreddit / title / selftext fields.
    :return:
    """
    instance = dict(flask.request.json)

    if instance['target'] != 'advice':
        return flask.jsonify({
            'instance': instance,
            'gen': 'Error',
        }), 200
    instances = instance['instances']

    model_size = instance.get('model_size', args.size)
    advices = load_estimator_and_predict_items(instances, date=datetime.utcnow(), model_size=model_size)
    instance['advice'] = advices
    with open(f'log.jsonl', 'a+') as logfile:
        logfile.write(json.dumps(instance) + '\n')

    return flask.jsonify({
        'gens': advices,
    }), 200

@click.command()
def serve():
    """Serve predictions on port 5000."""
    logging.basicConfig(
        format='%(asctime)s:%(levelname)s:%(name)s:%(message)s',
        level=logging.INFO)
    logger.info('Running prod server on http://127.0.0.1:5000/')


WSGIServer(('0.0.0.0', 5000), app).serve_forever()
