import os
from functools import partial

import tensorflow as tf
import mesh_tensorflow.transformer.dataset as transformer_dataset

SELFTEXT_DESIRED_LEN = 1250
SEQUENCE_LENGTH = {"inputs": 1280, "targets": 512}
_MONTHS = [
    'January', 'February', 'March', 'April', 'May', 'June', 'July', 'August',
    'September', 'October', 'November', 'December'
]
DATASET_IS_PACKED = False
# Data file constants
SPLITS = ["train", "val", "test"]
TSV_PATH = os.path.join(
    os.path.dirname(__file__),
    "{dataset_id}",
    "{split}_str.tsv"
)
TSV_COLNAMES = ["inputs", "targets1", "targets2"]
LOCAL_TFRECORDS_PATH = os.path.join(
    os.path.dirname(__file__),
    "{dataset_id}",
    "{split}.tfrecords"
)
GCS_TFRECORDS_PATH = "gs://seri2021-advice/turingadvice/reward/comparative/data/{dataset_id}/{split}.tfrecords"

def get_dataset(dataset_id, split, from_local=False):
    if from_local:
        tfrecords_path = LOCAL_TFRECORDS_PATH.format(split=split, dataset_id=dataset_id)
    else:
        tfrecords_path = GCS_TFRECORDS_PATH.format(split=split, dataset_id=dataset_id)
    serialized_dataset = tf.data.TFRecordDataset(tfrecords_path)
    feature_description = {
        "inputs": tf.io.FixedLenFeature(
            SEQUENCE_LENGTH["inputs"],
            tf.int64,
            default_value=[0]*SEQUENCE_LENGTH["inputs"]
        ),
        "inputs_position": tf.io.FixedLenFeature(
            SEQUENCE_LENGTH["inputs"],
            tf.int64,
            default_value=[0]*SEQUENCE_LENGTH["inputs"]
        ),
        "targets1": tf.io.FixedLenFeature(
            SEQUENCE_LENGTH["targets"],
            tf.int64,
            default_value=[0]*SEQUENCE_LENGTH["targets"]
        ),
        "targets1_position": tf.io.FixedLenFeature(
            SEQUENCE_LENGTH["targets"],
            tf.int64,
            default_value=[0]*SEQUENCE_LENGTH["targets"]
        ),
        "targets2": tf.io.FixedLenFeature(
            SEQUENCE_LENGTH["targets"],
            tf.int64,
            default_value=[0]*SEQUENCE_LENGTH["targets"]
        ),
        "targets2_position": tf.io.FixedLenFeature(
            SEQUENCE_LENGTH["targets"],
            tf.int64,
            default_value=[0]*SEQUENCE_LENGTH["targets"]
        ),
        "inputs_segmentation": tf.io.FixedLenFeature(
            SEQUENCE_LENGTH["inputs"],
            tf.int64,
            default_value=[0]*SEQUENCE_LENGTH["inputs"]
        ),
        "targets1_segmentation": tf.io.FixedLenFeature(
            SEQUENCE_LENGTH["targets"],
            tf.int64,
            default_value=[0]*SEQUENCE_LENGTH["targets"]
        ),
        "targets2_segmentation": tf.io.FixedLenFeature(
            SEQUENCE_LENGTH["targets"],
            tf.int64,
            default_value=[0]*SEQUENCE_LENGTH["targets"]
        )
    }
    tokens_dataset = serialized_dataset.map(
        lambda x: tf.io.parse_single_example(x, feature_description)
    )
    stacked_dataset = tokens_dataset.map(_stack_answer_pairs)
    return stacked_dataset

def _stack_answer_pairs(sample, concat=True):
    stack_fn = tf.concat if concat else tf.stack
    targets = stack_fn([sample["targets1"], sample["targets2"]], axis=0)
    targets_position = stack_fn([sample["targets1_position"], sample["targets2_position"]], axis=0)
    targets_segmentation = stack_fn([sample["targets1_segmentation"], sample["targets2_segmentation"]], axis=0)
    stacked_sample = {
        "inputs": sample["inputs"],
        "inputs_position": sample["inputs_position"],
        "inputs_segmentation": sample["inputs_segmentation"],
        "targets": targets,
        "targets_position": targets_position,
        "targets_segmentation": targets_segmentation
    }
    return stacked_sample

from datetime import datetime
from data.to_tfrecord_t5 import encoder, _trim_to_desired_length, _fix_reddit_text

def preprocess(
    question: dict, answer: str, vocabulary,
    max_selftext_len: int = SELFTEXT_DESIRED_LEN
    ):
    """
    Args:
    question: dict
        Dictionary with keys "subreddit", "created_utc", "title", "selftext"
    answer: str
        Answer to the question
    vocabulary: vocabulary.Vocabulary
        Str-to-int tokenizer
    max_selftext_len: int
        Max question selftext character length
    """
    dt_date = datetime.utcfromtimestamp(question["created_utc"])
    str_date = \
        _MONTHS[dt_date.month - 1] + " {}, {}".format(dt_date.day, dt_date.year)
    str_question = "Subreddit: {} Date: {} Title: {} Selftext: {}".format(
        _fix_reddit_text(question["subreddit"]),
        _fix_reddit_text(str_date),
        _fix_reddit_text(question["title"]),
        _fix_reddit_text(_trim_to_desired_length(
            encoder,
            question["selftext"],
            desired_len=max_selftext_len
        ))
    )
    str_answer = _fix_reddit_text(answer)
    tf_question = tf.cast(vocabulary.encode_tf(str_question), tf.int64)
    tf_question = tf.pad(
        tf_question,
        paddings=[[0, SEQUENCE_LENGTH["inputs"] - tf_question.shape[0]]]
    )
    tf_answer = tf.cast(vocabulary.encode_tf(str_answer), tf.int64)
    tf_answer = tf.pad(
        tf_answer,
        paddings=[[0, SEQUENCE_LENGTH["targets"] - tf_answer.shape[0]]]
    )
    return {"inputs": tf_question, "targets": tf_answer}