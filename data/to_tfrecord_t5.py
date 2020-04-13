# Let's just start with the tfrecord for Grover, just need to sub out tokenization
"""
Tokenization and tfrecord conversion for T5.

T5 uses a small vocab, so I had to remove symbols that are OOV, etc.
"""

import sys

sys.path.append('../')
import random
from datetime import datetime
from data.encoder import trim_paragraphs
from t5.data.sentencepiece_vocabulary import SentencePieceVocabulary
import sys
from unidecode import unidecode

import regex as re
from tqdm import tqdm
import json
import tensorflow as tf
import csv
import warnings

random.seed(123456)

# Make sure unidecode doesn't touch whatever is in the sentencepiece model
encoder = SentencePieceVocabulary(sentencepiece_model_file='gs://t5-data/vocabs/cc_all.32000/sentencepiece.model')
valid_symbols = sorted(set(''.join(encoder.decode([x]) for x in range(encoder.vocab_size))))

emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
with warnings.catch_warnings():
    old2new_unidecode = {}
    for i in range(sys.maxunicode):
        val_i = chr(i)
        if val_i in valid_symbols:
            continue
        try:
            unidecode_vali = unidecode(val_i)
        except Warning as e:
            # Surrogate character will be ignored?
            continue
        if unidecode_vali == val_i:
            continue
        if unidecode_vali == '[?]':
            old2new_unidecode[val_i] = ''
            continue
        if emoji_pattern.match(val_i) is not None:
            old2new_unidecode[val_i] = ''
            continue
        old2new_unidecode[val_i] = unidecode_vali
old2new_unidecode['\t'] = ' '
FAST_UNIDECODE = str.maketrans(old2new_unidecode)


def _fix_reddit_text(x):
    """TSV writer will complain if I can't do newlines. note that this will return an UNK"""
    x1 = x.translate(FAST_UNIDECODE)
    x3 = re.sub(r'[\s\n]*\n\n[\s\n]*', ' » ', x1, flags=re.MULTILINE)  # Double newline
    x4 = re.sub(r'[\s\n]*\n[\s\n]*', ' ', x3, flags=re.MULTILINE)  # Single newline
    x4 = re.sub(r'\s+', ' ', x4)
    return x4


def _trim_to_desired_length(encoder, text, desired_len=512):
    """ Trims a piece to the desired length, for sometimes long article pieces"""
    doc_len = len(encoder.encode(text))
    if doc_len <= desired_len:
        return text
    text = trim_paragraphs(selftext=text, num2del=1)
    return _trim_to_desired_length(encoder, text, desired_len=desired_len)


def tokenize_for_t5_advice_training(encoder, subreddit=None, date=None, title=None,
                                    selftext=None, body=None):
    """
    Tokenizes the post title / post selftext / comment body.
    If it's too long we'll cut some paragraphs at random from the selftext.

    :param subreddit: 'relationship_advice'
    :param date: datetime obj like datetime.datetime(2019, 7, 31, 23, 51, 21) always UTC time.
    :param title:
    :param selftext:
    :param body:
    :return:
    """
    if len(selftext) < 64:
        return None

    if len(body) < 64:
        return None

    article_pieces = {}
    if not isinstance(date, datetime):
        raise ValueError("Date must be a datetime obj. Provided {}".format(date))

    date_txt = ['January', 'February', 'March', 'April', 'May', 'June', 'July',
                'August', 'September', 'October', 'November', 'December'][date.month - 1] + ' {}, {}'.format(
        date.day, date.year)
    article_pieces['subreddit'] = subreddit
    article_pieces['date'] = date_txt
    article_pieces['title'] = title
    article_pieces['selftext'] = _trim_to_desired_length(encoder, selftext, desired_len=1250)
    article_pieces['body'] = body
    return {k: _fix_reddit_text(v) for k, v in article_pieces.items()}

if __name__ == '__main__':
    # Load from our static file
    training_examples = []
    advice = []
    with open('advice.jsonl', 'r') as f:
        for l in tqdm(f):
            item = json.loads(l)
            advice.append(item)
    training_examples_sorted = sorted(advice,
                                      key=lambda x: ({'test': 0, 'val': 1, 'train': 2}[x['split']], -x['created_utc']))

    num_test_remaining = 8192
    num_val_remaining = 8192
    num_entries = 0
    for x in tqdm(training_examples_sorted):

        budget = {'train': 10, 'test': num_test_remaining, 'val': num_val_remaining}[x['split']]

        x['tokens'] = []
        for comment in x['good_comments']:
            tokenized_comment = tokenize_for_t5_advice_training(
                encoder,
                date=datetime.utcfromtimestamp(x['created_utc']),
                subreddit=x['subreddit'],
                selftext=x['selftext'],
                title=x['title'],
                body=comment['body']
            )
            if tokenized_comment is not None:
                x['tokens'].append(tokenized_comment)
        x['tokens'] = x['tokens'][:budget]

        if num_entries < 10:
            for t in x['tokens']:
                print(x['tokens'], flush=True)

        num_entries += len(x['tokens'])
        if x['split'] == 'test':
            num_test_remaining = max(0, num_test_remaining - len(x['tokens']))
        if x['split'] == 'val':
            num_val_remaining = max(0, num_val_remaining - len(x['tokens']))

    random.shuffle(training_examples_sorted)

    # Change this file if you want to save somewhere else
    for split in ['train', 'val', 'test']:
        inferences_this_split = [y for x in training_examples_sorted if x['split'] == split for y in x['tokens']]
        with tf.io.gfile.GFile(f'{split}.tsv', 'w') as f:
            for item in inferences_this_split:
                f.write('\t'.join([item['subreddit'], item['date'], item['title'], item['selftext'], item['body']]) + '\n')