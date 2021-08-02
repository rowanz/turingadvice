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
from data.assertions import question_is_valid, answer_is_valid
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

def write_answer(question: dict, answer: dict, file):
    """
    Appends a line with the answer, including question context, to the file
    """
    tokenized_ans = tokenize_for_t5_advice_training(
        encoder,
        date=datetime.utcfromtimestamp(question["created_utc"]),
        subreddit=question["subreddit"],
        selftext=question["selftext"],
        title=question["title"],
        body=answer["body"]
    )
    if tokenized_ans:
        file.write(
            "\t".join([
                tokenized_ans['subreddit'],
                tokenized_ans['date'],
                tokenized_ans['title'],
                tokenized_ans['selftext'],
                tokenized_ans['body']]
            ) + "\n"
        )

if __name__ == '__main__':
    """
    Parameters
    ----------
    static_dataset_path : str
        Dataset generated by create_redditadvice_2019.py
    """
    TRAIN_ANSS_PER_Q = 10
    TOTAL_TEST_ANSS = 8192
    TOTAL_VAL_ANSS = 8192
    OUTPUT_TSV_PATH = "./data/{split}.tsv"
    static_dataset_path = sys.argv[1]
    # Which answers to use per question? Build anss_in_dataset
    valid_anss_per_question = []
    with open(static_dataset_path, "r") as static_dataset:
        print("Sorting questions by date")
        for line in tqdm(static_dataset):
            question = json.loads(line)
            if question_is_valid(question):
                continue
            else:
                valid_ans_ids = [
                    ans["id"] for ans in question["good_comments"]
                    if answer_is_valid(ans)
                ]
                valid_anss_per_question.append({
                    "q_id": question["id"],
                    "created_utc": question["created_utc"],
                    "split": question["split"],
                    "valid_ans_ids": valid_ans_ids
                })
    valid_anss_per_question = sorted(
        valid_anss_per_question, key=lambda x: x["created_utc"], reverse=True
    )
    anss_in_dataset = set() # set(ans_id)
    n_anss_left = {"test": TOTAL_TEST_ANSS, "val": TOTAL_VAL_ANSS}
    for valid_anss in valid_anss_per_question:
        if valid_anss["split"] == "train":
            anss_in_dataset.update(
                valid_anss["valid_ans_ids"][: ]
            )
        else:
            anss_in_dataset.update(
                valid_anss["valid_ans_ids"][:n_anss_left[valid_anss["split"]]]
            )
            n_anss_left[valid_anss["split"]] -= min(
                n_anss_left[valid_anss["split"]],
                len(valid_anss["valid_ans_ids"])
            )
    # Tokenize answers and write split dataset
    with open(static_dataset_path, "r") as static_dataset, \
        open(OUTPUT_TSV_PATH.format(split="train"), "w") as train_tsv,\
        open(OUTPUT_TSV_PATH.format(split="val"), "w") as val_tsv,\
        open(OUTPUT_TSV_PATH.format(split="test"), "w") as test_tsv:
        split_to_file = {
            "train": train_tsv,
            "val": val_tsv,
            "test": test_tsv
        }
        print("Writing file for each split")
        for line in tqdm(static_dataset):
            question = json.loads(line)
            for answer in question["good_comments"]:
                if answer["id"] in anss_in_dataset:
                    write_answer(
                        question=question,
                        answer=answer,
                        file=split_to_file[question["split"]]
                    )