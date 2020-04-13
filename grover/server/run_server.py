"""Serve predictions."""
import argparse
import json
import os

parser = argparse.ArgumentParser()

parser.add_argument('-gpu', type=int, default=0)
parser.add_argument('-size', type=str, default="mega")
parser.add_argument('-tag', type=str, default="")
parser.add_argument('-batch_size', type=int, default=1)

args = parser.parse_args()
GPUID = args.gpu
SIZE = args.size
TAG = "-" + args.tag if args.tag else ''

os.environ['CUDA_VISIBLE_DEVICES'] = str(GPUID)

import flask
from flask_cors import CORS
import tensorflow as tf
import sys

sys.path.append('../../')
from grover.lm.modeling import GroverConfig, sample_seq2seq
from data.encoder import get_encoder, extract_generated_target, _tokenize_reddit_post_pieces, trim_paragraphs
from data.tfrecord_utils import batch_index_iterator
import logging
from datetime import datetime
import click
from gevent.pywsgi import WSGIServer
import numpy as np
import pandas as pd

app = flask.Flask(__name__, template_folder='.')
CORS(app, resources={r'/api/*': {'origins': '*'}})

logger = logging.getLogger(__name__)

# SETUP
encoder = get_encoder()
news_config = GroverConfig.from_json_file(f'../lm/configs/{SIZE}.json')
batch_size = args.batch_size
top_p = 0.94

def _prepare_instance(instance, date, target='advice'):
    """
    Process each instance
    :param instance:
    :param target:
    :return:
    """
    if 'subreddit' not in instance:
        instance['subreddit'] = 'Advice'

    # Tokenize into pieces
    pieces = _tokenize_reddit_post_pieces(encoder, subreddit=instance['subreddit'], date=date,
                                          title=instance['title'],
                                          selftext=instance['selftext'] if target in ('selftext', 'advice') else '',
                                          max_date_length=5, max_subreddit_length=5, max_title_length=80,
                                          max_selftext_length=6000)
    # If too long, keep trimming the selftext
    while sum([len(x) for x in pieces]) > 1280:
        instance['selftext'] = trim_paragraphs(instance['selftext'], num2del=1)
        pieces['selftext'] = [encoder.begin_article] + encoder.encode(instance['selftext']) + [encoder.end_article]

    # OK NOW FORMAT CONTEXT
    instance['advice'] = ''
    if target == 'subreddit':
        context_formatted = [encoder.begin_domain]
        instance['date'] = ''
        instance['title'] = ''
        instance['selftext'] = ''
        eos_token_val = encoder.end_domain

    elif target == 'date':
        context_formatted = pieces['subreddit']
        context_formatted.append(encoder.begin_date)
        instance['selftext'] = ''
        instance['title'] = ''
        eos_token_val = encoder.end_date

    elif target == 'title':
        context_formatted = pieces['subreddit'] + pieces['date']
        context_formatted.append(encoder.begin_title)
        instance['selftext'] = ''
        eos_token_val = encoder.end_title

    elif target == 'selftext':
        context_formatted = pieces['subreddit'] + pieces['date'] + pieces['title']
        context_formatted.append(encoder.begin_article)
        eos_token_val = encoder.end_article

    elif target == 'advice':
        context_formatted = pieces['subreddit'] + pieces['date'] + pieces['title'] + pieces['selftext']
        context_formatted.append(encoder.begin_summary)
        eos_token_val = encoder.end_summary
    else:
        return None

    # Second sanity check
    if len(context_formatted) > 1280:
        context_formatted = context_formatted[-1280:]

    print("CTX is {}".format(encoder.decode(context_formatted)), flush=True)


    instance['eos_token'] = eos_token_val
    instance['context_formatted'] = context_formatted
    return instance


with tf.Session(config=tf.ConfigProto(allow_soft_placement=True), graph=tf.Graph()) as sess:
    initial_context = tf.placeholder(tf.int32, [batch_size, None])
    eos_token = tf.placeholder(tf.int32, [])
    ignore_ids = tf.placeholder(tf.bool, [news_config.vocab_size])

    tokens, probs = sample_seq2seq(news_config=news_config, initial_context=initial_context,
                           eos_token=eos_token, ignore_ids=ignore_ids, p_for_topp=top_p, max_len=1537)

    saver = tf.train.Saver()
    saver.restore(sess, f'ckpt-{SIZE}{TAG}/model.ckpt')

    # create a server endpoint to answer requests
    print("READY FOR GENERATION", flush=True)


    @app.route('/', methods=['GET'])
    def form_ask():
        """Return the demo page."""
        return flask.render_template('index.html', model_details="The model used is Grover {} with a max sequence length of 1536 tokens. It was trained on 600k reddit Q+A's for 20 epochs. The perplexity is {:.3f} on the val set.".format(args.size, {'mega': 12.56, 'large': 14.733, 'base': 17.565}[args.size]))


    @app.route('/api/ask', methods=['POST'])
    def api_ask():
        """Serve a prediction for a single instance."""
        instance = dict(flask.request.json)
        print("GOT A REQUEST for {}".format(instance), flush=True)

        target = instance.get('target', 'advice')
        instance = _prepare_instance(instance, date=datetime.utcnow(), target=target)
        if instance is None:
            return flask.jsonify({
                'instance': instance,
                'gen': 'error',
            }), 200

        eos_token_val = instance.pop('eos_token')
        context_formatted = instance.pop('context_formatted')

        # Indices we definitely DONT WANT TO PREDICT
        ignore_ids_np = np.array(encoder.special_tokens_onehot)
        ignore_ids_np[eos_token_val] = 0

        out = sess.run(tokens, feed_dict={initial_context: np.stack([context_formatted]*batch_size),
                                          eos_token: eos_token_val,
                                          ignore_ids: ignore_ids_np})

        out_decoded = extract_generated_target(
            output_tokens=out[0], encoder=encoder,
            target={'subreddit': 'domain', 'date': 'date',
                    'title': 'title', 'selftext': 'article', 'advice': 'summary'}[target])['extraction'].strip()
        print("SENDING BACK {}".format(out_decoded), flush=True)

        new_instance = {k: v for k, v in instance.items()}
        new_instance[target] = out_decoded
        new_instance['size'] = SIZE
        new_instance['tag'] = TAG.strip('-')
        new_instance['top_p'] = top_p

        with open(f'log{GPUID}.jsonl', 'a+') as logfile:
            logfile.write(json.dumps(new_instance) + '\n')

        return flask.jsonify({
            'instance': instance,
            'gen': out_decoded,
        }), 200

    @app.route('/api/askbatch', methods=['POST'])
    def api_askbatch():
        """
        instance has fields instances, each with subreddit / title / selftext fields.
        :return:
        """
        orig_instance = dict(flask.request.json)

        target = orig_instance.get('target', 'advice')
        date = datetime.utcnow()

        instances = [_prepare_instance(x, date=date, target=target) for x in orig_instance.pop('instances')]
        if any(x is None for x in instances):
            return flask.jsonify({
                'gens': 'error',
            }), 200

        eos_token_val = [x.pop('eos_token') for x in instances][0]
        ignore_ids_np = np.array(encoder.special_tokens_onehot)
        ignore_ids_np[eos_token_val] = 0

        things_to_process = pd.DataFrame(instances)
        things_to_process['ind'] = np.arange(len(instances))
        things_to_process['len'] = things_to_process['context_formatted'].apply(lambda x: len(x))
        things_to_process['out'] = ''
        things_to_process.sort_values(by='len', ascending=False, inplace=True)

        for b_start, b_end in batch_index_iterator(things_to_process.shape[0], batch_size=args.batch_size,
                                                   skip_end=False):
            these_ctx = things_to_process['context_formatted'].iloc[b_start:b_end].tolist()

            ctx_array = np.zeros((args.batch_size, max([len(x) for x in these_ctx])), dtype=np.int32) + encoder.padding
            for i, ctx_i in enumerate(these_ctx):
                ctx_array[i, :len(ctx_i)] = ctx_i

            out = sess.run(tokens, feed_dict={initial_context: ctx_array,
                                              eos_token: eos_token_val,
                                              ignore_ids: ignore_ids_np})
            for i, out_i in enumerate(out[:(b_end-b_start)]):
                # item = things_to_process.iloc[b_start + i]
                out_decoded = extract_generated_target(
                    output_tokens=out_i, encoder=encoder,
                    target={'subreddit': 'domain', 'date': 'date',
                            'title': 'title', 'selftext': 'article', 'advice': 'summary'}[target])['extraction'].strip()
                things_to_process.at[things_to_process.iloc[b_start + i].name, 'out'] = out_decoded

        # Sort back
        things_to_process.sort_values(by='ind', ascending=True, inplace=True)
        return flask.jsonify({
            'gens': things_to_process['out'].tolist(),
        }), 200


    @click.command()
    def serve():
        """Serve predictions on port 5000."""
        logging.basicConfig(
            format='%(asctime)s:%(levelname)s:%(name)s:%(message)s',
            level=logging.INFO)
        logger.info('Running prod server on http://127.0.0.1:5000/')


    WSGIServer(('0.0.0.0', 5000 + GPUID), app).serve_forever()
