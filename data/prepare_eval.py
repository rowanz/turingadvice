"""
Use this script to get all the model advice, and the reddit advice
"""
import sys

sys.path.append('../')
# Enable all logging
import logging
import argparse
from data.encoder import clean_reddit_text
import praw
from praw.models import MoreComments
from praw.models.reddit import submission, comment
from typing import Iterable, List
from collections import OrderedDict
from datetime import datetime
import pytz
import re
import mistune
from spacy.tokens import Token
from tqdm import tqdm
import random
from allennlp.common.util import get_spacy_model

spacy_model = get_spacy_model('en_core_web_sm', pos_tags=False, parse=False, ner=False)

parser = argparse.ArgumentParser()
parser.add_argument('-username', type=str, default="jarzebekz")
parser.add_argument('-budget', type=int, default=200)
parser.add_argument('-date_tag', type=str, default="Feb-14-20", help='Default date tag to use')

args = parser.parse_args()

handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)
logger = logging.getLogger('prawcore')
logger.setLevel(logging.DEBUG)
logger.addHandler(handler)
###

# You'll need to fill in these values
# reddit_api = praw.Reddit(client_id=reddit_acct['clientid'],
#                          client_secret=reddit_acct['secret'],
#                          username=reddit_acct['username'],
#                          password=reddit_acct['password'],
#                          user_agent=reddit_acct['useragent'])
reddit_api = None
if reddit_api is None:
    print("Add in your authentication here", flush=True)


def extract_info_from_sub(sub: submission.Submission):
    """
    Turns the submission into a dict of useful  info
    :param sub:
    :return:
    """
    x_info = OrderedDict()
    x_info['score'] = sub.score
    x_info['num_comments'] = sub.num_comments
    x_info['upvote_ratio'] = sub.upvote_ratio
    x_info['submitted_time'] = int(sub.created_utc)
    x_info['extracted_time'] = datetime.utcnow().timestamp()
    x_info['seconds_elapsed'] = (datetime.utcfromtimestamp(x_info['extracted_time']) -
                                 datetime.utcfromtimestamp(x_info['submitted_time'])).total_seconds()
    x_info['title'] = sub.title
    x_info['selftext'] = sub.selftext
    x_info['permalink'] = sub.permalink
    x_info['id'] = sub.id
    x_info['subreddit'] = sub.subreddit.display_name
    x_info['op'] = sub.author.name if sub.author is not None else None
    return x_info


def _contains_links(post):
    """
    Break if the post contains a URL to anything
    :param post:
    :return:
    """
    html_format = mistune.markdown(post)
    if '<a>' in html_format or 'http://' in html_format or 'https://' in html_format:
        return True
    return False


def _contains_bad_words(spacy_tokens: List[Token],
                        bad_words=(
                                'assault', 'assaulted', 'suicide', 'suicidal', 'prostitution', 'self harm', 'abuse'), ):
    text_simplified = ' '.join([x.orth_.lower() for x in spacy_tokens if x.is_alpha])
    found_words = re.search(r'\b(' + r'|'.join(bad_words) + r')\b', text_simplified)
    if found_words is not None:
        print("Skipping post bc {}".format(found_words.group(0)))
        return True
    return False


def _comment_refers_to_others(spacy_tokens: List[Token],
                              bad_words=('other commenters', 'top comment', 'the other comments',), ):
    text_simplified = ' '.join([x.orth_.lower() for x in spacy_tokens if x.is_alpha])
    found_words = re.search(r'\b(' + r'|'.join(bad_words) + r')\b', text_simplified)
    if found_words is not None:
        print("Skipping post bc {}".format(found_words.group(0)))
        return True
    return False


def iterate_toplvl_comments(sub: submission.Submission) -> Iterable[comment.Comment]:
    """
    Returns all top-level comments in a submission.
    :param sub: Reddit submission like reddit_api.submission(id="3g1jfi")
    :return: An iterable of top-level comments, praw.models.reddit.comment
    """

    def _iterate_helper(comment_list):
        for comment in comment_list:
            if comment.depth == 0:
                if isinstance(comment, MoreComments):
                    yield from _iterate_helper(comment.comments())
                else:
                    yield comment

    return _iterate_helper(sub.comments.list())


def _vis_time(dt: datetime):
    """
    :param dt: datetime obj
    :return: A string representation
    """
    pst_time = pytz.utc.localize(dt).astimezone(pytz.timezone('US/Pacific'))
    return pst_time.strftime('%A, %B %e, %Y %l:%M %p %Z')


def comment_to_info(com: comment.Comment):
    com_info = OrderedDict()
    com_info['id'] = com.id
    com_info['score'] = com.score
    com_info['submitted_time'] = com.created_utc
    com_info['body'] = com.body
    com_info['author'] = com.author.name if com.author is not None else None
    return com_info


QTYPE_MAP = {'relationships': 'Relationship',
             'relationship_advice': 'Relationship',
             'dating_advice': 'Relationship',
             'legaladvice': 'Legal',
             # 'NoStupidQuestions': 16093,
             'Advice': 'Life',
             'internetparents': 'Life',
             'dating': 'Relationship',
             'needadvice': 'Life',
             'techsupport': 'Technology',
             'Marriage': 'Relationship',
             'love': 'Relationship'}


def get_posts_from_reddit(max_days_elapsed=14, post_limit=2000):
    """
    Iterates through posts on reddit
    :param max_days_elapsed:
    :param post_limit:
    :return:
    """
    # from collections import defaultdict
    # import json
    # subreddit_weights = defaultdict(int)
    # with open('advice.jsonl', 'r') as f:
    #     for l in f:
    #         item = json.loads(l)
    #         subreddit_weights[item['subreddit']] += 1
    subreddit_weights = {'relationships': 84960,
                         'relationship_advice': 24140,
                         'dating_advice': 4288,
                         'legaladvice': 47421,
                         # 'NoStupidQuestions': 16093,
                         'Advice': 3998,
                         'internetparents': 1594,
                         'dating': 1170,
                         'needadvice': 1350,
                         'techsupport': 2216,
                         'Marriage': 1297,
                         'love': 93}
    # Oversample a bit, at least for now
    for x in tqdm(reddit_api.subreddit('+'.join(subreddit_weights)).top('month' if max_days_elapsed > 7 else 'week',
                                                                        limit=post_limit), total=post_limit):
        # Skip comments for now
        x.comment_limit = 0
        x.comment_sort = 'top'

        x_info = extract_info_from_sub(x)

        # At least 36 hours ago, + upvote ratio
        if (x_info['seconds_elapsed'] < 36 * 60 * 60) or (x_info['upvote_ratio'] < 0.5) or \
                (x_info['seconds_elapsed'] > max_days_elapsed * 60 * 60 * 24):
            # print("Skipping >36hrs ago", flush=True)
            continue

        # Skip if the title contains UPDATE, also skip meta
        if (re.search(r'\W*update[\W\b]', x_info['title'].lower()) is not None) or (
                re.search(r'Update|UPDATE|META', x_info['title']) is not None) or x.pinned or x.stickied:
            # print("Contains update", flush=True)
            continue

        if _contains_links(x_info['selftext']):
            # print("Contains links", flush=True)
            continue

        # Sanity check
        if x_info['score'] < 20:
            # print("Skipping loscore", flush=True)
            continue

        # Remove updates / 'edited' msgs.
        x_info['selftext'] = clean_reddit_text(x_info['selftext'])

        # Remove non-questions
        if ('?' not in x_info['title']) and x_info['subreddit'] in ('dating', 'Marriage', 'love', 'dating_advice'):
            continue

        spacy_tokens_ctx = [x for x in spacy_model('{} {}'.format(x_info['title'], x_info['selftext']))]

        if _contains_bad_words(spacy_tokens_ctx):
            # print("Skipping badwords", flush=True)
            continue

        # print("Ctx is {} tokens".format(len(spacy_tokens_ctx)), flush=True)
        if (len(spacy_tokens_ctx) < 128) or (len(spacy_tokens_ctx) > 1280):
            # print("Skipping CTX", flush=True)
            continue

        # Now get the top comments
        x.comment_limit = 16
        good_comments = []
        all_comments = []
        for c in iterate_toplvl_comments(x):
            all_comments.append(c)
            if c.score_hidden or (c.author is None) or (
                    c.author.name in (
                    x_info['op'], 'AutoModerator')) or c.score < 20 or c.is_submitter or c.stickied or c.edited:
                continue
            c_info = comment_to_info(c)
            c_info['body'] = clean_reddit_text(c_info['body'])

            spacy_tokens_com = [x for x in spacy_model(c_info['body'])]
            if len(spacy_tokens_com) < 32:
                continue

            if _comment_refers_to_others(spacy_tokens_com):
                continue

            good_comments.append(c_info)
            # FOUND
            break

        if len(good_comments) == 0:
            # print("Skipping len(good_comments) == 0", flush=True)
            continue
        best_advice = good_comments[0]

        for k, v in best_advice.items():
            x_info[f'bestadvice_{k}'] = v

        yield x_info


info = [x for x in get_posts_from_reddit(post_limit=1000)]

random.shuffle(info)

# I used 200/1000
info = info[:200]
