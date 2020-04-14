import argparse
import json
import os

parser = argparse.ArgumentParser()
parser.add_argument('-gpu', type=int, default=2)

args = parser.parse_args()
GPUID = args.gpu
os.environ['CUDA_VISIBLE_DEVICES'] = str(GPUID)

import flask
from flask_cors import CORS
import tensorflow as tf
import sys

import logging
from datetime import datetime
import click
from gevent.pywsgi import WSGIServer
import numpy as np
import pandas as pd
from allennlp.common.util import get_spacy_model
import mistune
import os
from google.cloud import storage
from spacy.tokens import Token
import boto3
import json
import demoji
import mistletoe
from tqdm import tqdm
import pandas as pd
import random
from copy import deepcopy
import scipy.sparse
from collections import defaultdict
app = flask.Flask(__name__, template_folder='.')
CORS(app, resources={r'/api/*': {'origins': '*'}})

logger = logging.getLogger(__name__)


spacy_model = get_spacy_model('en_core_web_sm', pos_tags=False, parse=False, ner=False)

print("You need to have the file redditadvice2019.jsonl in your data/ directory.", flush=True)
if os.path.exists('counts.npz'):
    print("Loading from CACHE!", flush=True)
    tfidf_coo = scipy.sparse.load_npz('counts.npz')

    # Compute the denominator
    tfidf_coo_denom = np.sqrt(tfidf_coo.power(2).dot(np.ones(tfidf_coo.shape[1], dtype=np.float16)))
    #
    # tfidf_coo_denom = np.load('counts_denom.npy')
    idf = np.load('idf.npy')
    advice = []
    with open('../data/redditadvice2019.jsonl', 'r') as f:
        for l in tqdm(f, total=188620):
            item = json.loads(l)
            if len(item['good_comments']) == 0:
                continue
            advice.append(item)

    with open('word_to_idx.json', 'r') as f:
        word_to_idx = json.load(f)
    idx_to_word = [w for w, i in sorted(word_to_idx.items(), key=lambda x: x[1])]

else:
    advice = []
    word2count = defaultdict(int)
    word2count_doc = defaultdict(int)
    with open('../data/redditadvice2019.jsonl', 'r') as f:
        for l in tqdm(f, total=188620):
            item = json.loads(l)
            item['ctx_tokenized'] = [x.lemma_.lower() for x in spacy_model('{} {}'.format(item['title'], item['selftext']))]
            for tok in item['ctx_tokenized']:
                word2count[tok] += 1
            for tok in set(item['ctx_tokenized']):
                word2count_doc[tok] += 1
            if len(item['good_comments']) == 0:
                continue
            advice.append(item)

    print("Making vocabulary + IDF", flush=True)
    idx_to_word = ['UNK'] + [x[0] for x in sorted(word2count.items(), key=lambda x: -x[1]) if x[1] >= 10]
    word_to_idx = {w: i for i, w in enumerate(idx_to_word)}
    idf = np.zeros(len(word_to_idx), dtype=np.float32)
    for idx, word in enumerate(idx_to_word):
        if idx == 0:
            idf[0] = np.log(1.01)
            continue
        idf[idx] = np.log(188620 / (1.0 + word2count_doc.get(word, 1.0)))

    # def ctx_tokenized_to_vec(toks):
    #     vec = np.zeros(len(word_to_idx), dtype=np.float32)
    #     # toks = [x.orth_.lower() for x in spacy_model(ctx)]
    #     for tok in toks:
    #         vec[word_to_idx.get(tok, 0)] += 1.0
    #     vec /= len(toks)
    #     return vec * idf

    print("Turning everything into the count matrix", flush=True)
    tfidf = scipy.sparse.dok_matrix((len(advice), len(word_to_idx)), dtype=np.float16)
    for i, item in enumerate(tqdm(advice)):

        ind_to_count = defaultdict(int)
        for tok in item['ctx_tokenized']:
            ind_to_count[word_to_idx.get(tok, 0)] += 1
        for ind, count in sorted(ind_to_count.items()):
            tfidf[i, ind] = count * idf[ind] / len(item['ctx_tokenized'])
    tfidf_coo = scipy.sparse.coo_matrix(tfidf)

    print("DUMPING TO FILE", flush=True)
    with open('word_to_idx.json', 'w') as f:
        json.dump(word_to_idx, f)
    scipy.sparse.save_npz('counts.npz', tfidf_coo)
    np.save('idf.npy', idf)



print("READY TO GO!", flush=True)
def gen_advice(item):
    item_tokenized = [x.lemma_.lower() for x in spacy_model('{} {}'.format(item['title'], item['selftext']))]
    item_vec = np.zeros(len(word_to_idx), dtype=np.float16)
    ind_to_count = defaultdict(int)
    for tok in item_tokenized:
        ind_to_count[word_to_idx.get(tok, 0)] += 1
    for ind, count in sorted(ind_to_count.items()):
        item_vec[ind] = count * idf[ind] / len(item_tokenized)
    sim = tfidf_coo.dot(item_vec) + (1e-5) * np.random.rand(len(advice))
    # Divide by cosine similarity denominator, other thing is a constant
    sim /= tfidf_coo_denom
    most_similar = advice[int(np.argmax(sim))]
    print('https://reddit.com/r/{}/comments/{}/_/{}/'.format(most_similar['subreddit'], most_similar['id'], most_similar['good_comments'][0]['id']), flush=True)
    return most_similar['good_comments'][0]['body']

@app.route('/', methods=['GET'])
def form_ask():
    """Return the demo page."""
    return flask.render_template('index.html', model_details="Retrieval + BoW system")


@app.route('/api/ask', methods=['POST'])
def api_ask():
    """Serve a prediction for a single instance."""
    instance = dict(flask.request.json)
    print("GOT A REQUEST for {}".format(instance), flush=True)

    gen = gen_advice(instance)
    return flask.jsonify({
        'instance': instance,
        'gen': gen,
    }), 200

@app.route('/api/askbatch', methods=['POST'])
def api_askbatch():
    """
    instance has fields instances, each with subreddit / title / selftext fields.
    :return:
    """
    orig_instance = dict(flask.request.json)
    return flask.jsonify({
        'gens': [gen_advice(x) for x in orig_instance.pop('instances')],
    }), 200


@click.command()
def serve():
    """Serve predictions on port 5000."""
    logging.basicConfig(
        format='%(asctime)s:%(levelname)s:%(name)s:%(message)s',
        level=logging.INFO)
    logger.info('Running prod server on http://127.0.0.1:5000/')


WSGIServer(('0.0.0.0', 5003), app).serve_forever()
