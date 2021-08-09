import os
from warnings import warn

import tensorflow as tf
from mesh_tensorflow.transformer.dataset import trim_and_pad_dataset

from t5.data.utils import _SHUFFLE_BUFFER_SIZE, encode_string_features
from data.to_tfrecord_t5 import encoder as TOKENIZER

SELFTEXT_DESIRED_LEN = 1250
SEQUENCE_LENGTH = {"inputs": 1280, "targets": 512}
_MONTHS = [
    'January', 'February', 'March', 'April', 'May', 'June', 'July', 'August',
    'September', 'October', 'November', 'December'
]
DATASET_IS_PACKED = False
# Data file constants
SPLITS = ["train", "val", "test"]
LOCAL_TSV_PATH = os.path.join(
    os.path.dirname(__file__),
    "{dataset_id}/{split}_str.tsv"
)
GCS_TSV_PATH = "gs://{bucket_name}/turingadvice/reward/comparative/data/{dataset_id}/{split}_str.tsv"
TSV_COLNAMES = ["inputs", "targets1", "targets2"]
LOCAL_TFRECORDS_PATH = os.path.join(
    os.path.dirname(__file__),
    "{dataset_id}/{split}.tfrecords"
)
GCS_TFRECORDS_PATH = "gs://{bucket_name}/turingadvice/reward/comparative/data/{dataset_id}/{split}.tfrecords"

def get_dataset(
    bucket_name, dataset_id, split, from_local=False, from_tfrecords=False,
    stack_answer_pairs=True, shuffle_buffer_size=10000
    ):
    dir_params = {
        "bucket_name": bucket_name,
        "split": split,
        "dataset_id":dataset_id
    }
    if from_tfrecords:
        assert stack_answer_pairs, "Unstacked answer pairs unavailable when from_tfrecords=True"
        if from_local:
            tfrecords_path = LOCAL_TFRECORDS_PATH.format(**dir_params)
        else:
            tfrecords_path = GCS_TFRECORDS_PATH.format(**dir_params)
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
        padded_dataset = serialized_dataset.map(
            lambda x: tf.io.parse_single_example(x, feature_description)
        )
        unshuffled_dataset = padded_dataset.map(_stack_answer_pairs)
    else:
        if from_local:
            tsv_path = LOCAL_TSV_PATH.format(**dir_params)
        else:
            tsv_path = GCS_TSV_PATH.format(**dir_params)
        tsv_dataset = tf.data.experimental.CsvDataset(
            tsv_path,
            record_defaults=["", "", ""],
            field_delim="\t",
            use_quote_delim=False
        )
        tsv_dataset = tsv_dataset.map(
            lambda *x: {c: x[i] for i, c in enumerate(TSV_COLNAMES)}
        )
        tokens_dataset = encode_string_features(
            dataset=tsv_dataset,
            vocabulary=TOKENIZER,   
            copy_plaintext=False,
            keys=TSV_COLNAMES
        )
        unpadded_dataset = tokens_dataset.map(_add_position_and_segmentation)
        SEQUENCE_LENGTH.update({
            "targets1": SEQUENCE_LENGTH["targets"],
            "targets2": SEQUENCE_LENGTH["targets"],
        })
        _sequence_length = {
            **SEQUENCE_LENGTH,
            **{k + "_position": v for k, v in SEQUENCE_LENGTH.items()},
            **{k + "_segmentation": v for k, v in SEQUENCE_LENGTH.items()}
        }
        padded_dataset = trim_and_pad_dataset(
            dataset=unpadded_dataset,
            length=_sequence_length
        )
        if stack_answer_pairs:
            unshuffled_dataset = padded_dataset.map(_stack_answer_pairs)
        else:
            unshuffled_dataset = padded_dataset
    # Shuffle dataset
    if shuffle_buffer_size > 0:
        shuffled_dataset = unshuffled_dataset.shuffle(
            buffer_size = shuffle_buffer_size,
            seed = 41
        )
        return shuffled_dataset
    else:
        return unshuffled_dataset

def _add_position_and_segmentation(sample):
    """
    These tensors are generated by mtf.transformer.dataset.pack_or_pad with
    pack=True. We're not supporting packing, so we have to add them manually
    for mtf transformers to work.
    """
    new_items = {}
    for k, v in sample.items():
        new_items[k + "_segmentation"] = (v * 0) + 1
        new_items[k + "_position"] = tf.range(len(v))
    sample.update(new_items)
    return sample

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

def get_checkpoint_paths(base_dir, min_steps=-1):
    """
    Args:
    base_dir: str
        Directory where we'll look for checkpoints
    min_steps: int
        Ignore checkpoints with less than these many steps
    
    Returns:
    ckpt_paths: [str]
        List of absolute checkpoint paths
    """
    index_paths = tf.io.gfile.glob(f"{base_dir}/model.ckpt-[0-9]*.index")
    ckpt_paths = [p.replace(".index", "") for p in index_paths]
    filtered_ckpt_paths = []
    for ckpt_path in ckpt_paths:
        try:
            ckpts_steps = int(ckpt_path.split("-")[-1])
            if ckpts_steps >= min_steps:
                filtered_ckpt_paths.append(ckpt_path)
        except:
            warn(f"Ignoring unparseable checkpoint path: {ckpt_path}")
    return filtered_ckpt_paths

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