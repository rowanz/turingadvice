"""
Fields
posts
 created_utc,subreddit,author,num_comments,score,title,selftext,id,gilded,retrieved_on

comments
 created_utc,subreddit,author,parent_id,link_id,score,body,id,gilded,retrieved_on
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


def _load_item(l):
    """ fixes up a few things, like makes sure things are strings"""
    item = json.loads(l)
    item['score'] = int(item['score'])
    item['created_utc'] = int(item['created_utc'])
    if 'retrieved_on' in item:
        item['retrieved_on'] = int(item['retrieved_on'])

    item['gilded'] = True if int(item['gilded']) == 1 else False
    if 'num_comments' in item:
        item['num_comments'] = int(item['num_comments'])

    # Everything seems to start with t3_
    if 'link_id' in item:
        item['link_id'] = item['link_id'][3:]
    if 'parent_id' in item:
        item['parent_id'] = item['parent_id'][3:]

    # Fix edit messages, I think this works
    for comment_field in 'selftext', 'body':
        if comment_field in item:
            old_txt = item[comment_field]
            item[comment_field] = clean_reddit_text(item[comment_field])

            # For debugging.
            # if item[comment_field] != old_txt:
            #     print("\n\nURL {}\nOriginal {}:\n{}\n~~~\nNew {}:\n{}\n=======\n\n".format(
            #        'https://reddit.com/r/{}/comments/{}/'.format(item['subreddit'], item.get('link_id', item['id'])),
            #         comment_field,
            #         old_txt,
            #         comment_field,
            #         item[comment_field],
            #     ), flush=True)

    return item


print("POSTS", flush=True)
postid_to_post = {}
postid_to_comments = defaultdict(list)
with open('/home/rowan/datasets2/redditscraper/posts-advice.jsonl', 'r') as f:
    for l in tqdm(f):
        item = _load_item(l)

        if 'META' in item['title']:
            continue

        # It's probably OK if updates are included, but maybe we won't respond to them in testing? idk.
        # if item['title'].lower().startswith(('update', '[update]', '(update)', '“update')) or 'UPDATE' in item['title']:
        #     continue

        # if 'update' in item['title'].lower():
        #     continue

        html_format = mistune.markdown(item['selftext'])
        if '<a>' in html_format or 'http://' in html_format or 'https://' in html_format:
            continue

        postid_to_post[item['id']] = item

print("COMMENTS", flush=True)
with open('/home/rowan/datasets2/redditscraper/comments-advice.jsonl', 'r') as f:
    for l in tqdm(f):
        item = _load_item(l)
        postid_to_comments[item['link_id']].append(item)


def merge_post_with_comments(post_id):
    """
    Connects posts with good comments.
    :param post_id: The post ID
    :return: None if not found else an item
    """
    post = postid_to_post.get(post_id, None)
    if post is None:
        return None
    top_lvl_comments = [x for x in postid_to_comments[post_id] if x['parent_id'] == x['link_id']]
    if len(top_lvl_comments) == 0:
        return None

    highest_scoring_toplvl_comment = max(top_lvl_comments, key=lambda x: x['score'])
    if highest_scoring_toplvl_comment['score'] < 20:
        return None

    # Return everything that has a score of at least 1/10 of the best one, while having >10 karma, or is gilded.
    # Remove duplicates if they exist.
    good_comments = []
    seen_comment_ids = set()
    for x in top_lvl_comments:
        if x['id'] in seen_comment_ids:
            continue

        if x['score'] > highest_scoring_toplvl_comment['score'] / 10.0 or x['gilded'] == 1:
            good_comments.append(x)
            seen_comment_ids.add(x['id'])

    return_dict = {k: v for k, v in post.items()}
    return_dict['good_comments'] = good_comments
    return return_dict


training_examples = [merge_post_with_comments(id) for id in tqdm(postid_to_post)]
training_examples = [x for x in training_examples if x is not None]

# Sort from new -> old.
# Could use datetime.utcfromtimestamp
training_examples_sorted = sorted(training_examples, key=lambda x: -x['created_utc'])

print("TOKENIZING AND TRIMMING", flush=True)
num_test = 8192
num_val = 8192
num_entries = 0
for x in tqdm(training_examples_sorted):
    if num_entries < num_test:
        x['split'] = 'test'
        budget = num_test - num_entries
    elif num_entries < (num_test + num_val):
        x['split'] = 'val'
        budget = num_test + num_val - num_entries
    else:
        x['split'] = 'train'
        budget = 10

    x['tokens'] = []
    for comment in x['good_comments']:
        tokenized_comment = tokenize_for_grover_advice_training(
            encoder,
            date=datetime.utcfromtimestamp(x['created_utc']),
            subreddit=x['subreddit'],
            selftext=x['selftext'],
            title=x['title'],
            body=comment['body'],
            desired_len=1536)
        if tokenized_comment is not None:
            x['tokens'].append(tokenized_comment)
    x['tokens'] = x['tokens'][:budget]
    num_entries += len(x['tokens'])

lowest_val_utc = min([x['created_utc'] for x in training_examples_sorted if x['split'] == 'val'])
print("Val starts {}".format(datetime.utcfromtimestamp(lowest_val_utc)), flush=True)
lowest_test_utc = min([x['created_utc'] for x in training_examples_sorted if x['split'] == 'test'])
print("Test starts {}".format(datetime.utcfromtimestamp(lowest_test_utc)), flush=True)
highest_test_utc = max([x['created_utc'] for x in training_examples_sorted if x['split'] == 'test'])
print("Test ends {}".format(datetime.utcfromtimestamp(highest_test_utc)), flush=True)

random.shuffle(training_examples_sorted)

# Cache to a static file
with open('advice.jsonl', 'w') as f:
    for item in training_examples_sorted:
        f.write(json.dumps(item) + '\n')

# lens = [len(encoder.encode('{} {} {}'.format(x['selftext'], x['title'], x['good_comments'][0]['body']))) for x in tqdm(training_examples)]
# Create train / test / val split.

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