"""
"""

import json
import sys
from collections import defaultdict, OrderedDict

from tqdm import tqdm

sys.path.append('../')
from data.encoder import get_encoder, clean_reddit_text, tokenize_for_grover_advice_training
from data.tfrecord_utils import S3TFRecordWriter, int64_list_feature
import mistune
from datetime import datetime
import random
import tensorflow as tf

encoder = get_encoder()
random.seed(123456)
advice = []
with open('advice.jsonl', 'r') as f:
    for l in tqdm(f):
        item = json.loads(l)
        advice.append(item)
training_examples_sorted = sorted(advice,
                                  key=lambda x: ({'test': 0, 'val': 1, 'train': 2}[x['split']], -x['created_utc']))

random.shuffle(training_examples_sorted)

for split in ['train', 'val', 'test']:
    inferences_this_split = [y for x in training_examples_sorted if x['split'] == split for y in x['tokens']]

    num_folds = 32 if split == 'train' else 1

    print("{} inferences for {}".format(len(inferences_this_split), split))
    for fold in range(num_folds):

        # Change this file if you want to save somewhere else
        file_name = '{}{:02d}of{}.tfrecord'.format(split, fold, num_folds)
        with S3TFRecordWriter(file_name) as writer:
            for i, item in enumerate(tqdm(inferences_this_split)):
                if i % num_folds == fold:
                    features = OrderedDict()
                    features['context'] = int64_list_feature(item['context'])
                    features['target'] = int64_list_feature(item['target'])
                    ex = tf.train.Example(features=tf.train.Features(feature=features))
                    writer.write(ex.SerializeToString())
