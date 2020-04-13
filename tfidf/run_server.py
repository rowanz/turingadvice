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

# sys.path.append('../../')
# from grover.lm.modeling import GroverConfig, sample_seq2seq
# from data.encoder import get_encoder, extract_generated_target, _tokenize_reddit_post_pieces, trim_paragraphs
# from data.tfrecord_utils import batch_index_iterator
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




# selftext = """Tldr: my friend is acting like a clingy girlfriend and it's getting to be too much.
#
# Weird thing to be unsure of, I know. I'm straight. So there is the first wrench in that. She has been acting very weird lately. She's always asking me to hold stuff for her, has lately been using that "cutsey" voice girls use when they want something, she whines and whines when I tell her no to something (like walking her to her dorm or not eating dinner with her), when we went to a store she asked me 4 times to buy her something and when I told her no she pouted about it. Another time at a store, I dropped 20 bucks on her because she wanted a bracelet making kit and she said she would get me back (she hasn't)
#
# The thing that made me notice this trend was today. I made plans to go out with this guy I've been talking to for Valentines day. She called me (15 minutes before class) I told her that I had to go and she asked why and I said I wanted to call the guy really quickly because he wasn't in class earlier. She started whining and being really annoying about it. Just yelling "oh I cant believe you would hang up on me to talk to [guy]" (She did it in the joking exaggerated way that we sometimes talk) but the thing is.... she wouldn't let me hang up. " I will be so mad if you hang up on me to talk to [guy]" she literally went on and on about me hanging up on her to talk to him BUT I HADN'T EVEN HUNG UP ON HER. Then I told her I needed to pee before class "Oh then take me into the bathroom with you." Class is starting I have to go "Class doesn't start for 3 minutes! Stay on with me!" Just nonstop whining. I hung up when I got to the toilet and she called 3 times in a row. I didn't get to call the guy until after class.
#
# Then when I had a meeting tonight I was early so I could write. She called me and asked if I could eat with her. I said no, I have a meeting. Again, with the whining and being loud while my supervisors had walked into the room. I cant imagine what I must have sounded like, trying to reason with her about having to stay for a meeting.
#
# She steals my things and expects me to chase her, she lays on my bed (i dont like when people are on my things) when she is in my room. And she calls me by a name that she knows I don't like purposely (at this point I've just ignored getting upset about this. If I get riled up about it, she keeps going)
#
# So uhmmmm.... help?
# """
#
spacy_model = get_spacy_model('en_core_web_sm', pos_tags=False, parse=False, ner=False)

if os.path.exists('counts.npz'):
    print("Loading from CACHE!", flush=True)
    tfidf_coo = scipy.sparse.load_npz('counts.npz')

    # Compute the denominator
    tfidf_coo_denom = np.sqrt(tfidf_coo.power(2).dot(np.ones(tfidf_coo.shape[1], dtype=np.float16)))
    #
    # tfidf_coo_denom = np.load('counts_denom.npy')
    idf = np.load('idf.npy')
    advice = []
    with open('../data/advice.jsonl', 'r') as f:
        for l in tqdm(f, total=188620):
            item = json.loads(l)
            if len(item['good_comments']) == 0:
                continue
            advice.append(item)

    with open('word_to_idx.json', 'r') as f:
        word_to_idx = json.load(f)
    idx_to_word = [w for w, i in sorted(word_to_idx.items(), key=lambda x: x[1])]

else:
    print("LOADING ADVICE.JSONL", flush=True)
    print("You need to have the file advice.jsonl in your data/ directory.")
    advice = []
    word2count = defaultdict(int)
    word2count_doc = defaultdict(int)
    with open('../data/advice.jsonl', 'r') as f:
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


# {'subreddit': 'subreddit', 'title': 'title', 'selftext': 'selftext', 'advice': 'You should buy a dog.', 'target': 'title'}

print("READY TO GO!", flush=True)
# item = {'subreddit': 'relationships',
#  'title': 'I [29M] with my wife [29F] are visiting our home town with our newborn baby. How do I tell my parents [63M and 56F] that I can’t bring my son into their house?',
#  'selftext': """Okay, so this has been as issue my wife and I have tip toed around for a while. I love my parents, we both do, but my dad has spent his entire life smoking cigarettes. He smokes in the house, in his car, almost nonstop.
#
# I’ve accepted that he has a serious addiction and he’s the only one who can change that. BUT he smokes with zero ventilation, has for years. The house gives me a headache from the smoke. My wife has asthma and I’ve taken her to the ER the last two times she’s been because she had attacks. Even my parents clothes smell like smoke. I’ve told my mom, she acts like she forgets every time and that nothing is wrong. My dad gets angry whenever you bring up his smoking (because he doesn’t want to admit the problem it is, I assume).
#
# We were going to explain this to my parents when my wife was pregnant last time we visited town, but my mom got very sick and went to the hospital for some days and bringing it up just didn’t seem like the time since she nearly died.
#
# Now they want us to pick a night for us to come over and invite family, but there’s no way I can feel comfortable bringing my son or wife in the house.
#
# How can I go about explaining to them the seriousness of this issue so they won’t just brush it off?
#
# TL;DR I don’t want to expose my son to my parents smokey house
# """}

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
