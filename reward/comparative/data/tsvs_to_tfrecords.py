import os
from functools import partial

import tensorflow as tf
from mesh_tensorflow.transformer.dataset import pack_or_pad

from data.to_tfrecord_t5 import encoder as TOKENIZER
from reward.comparative.data.jsonl_to_tsvs import TSV_PATH, TSV_COLNAMES
from t5.data.utils import encode_string_features

TFRECORDS_PATH = os.path.dirname(__file__) + "/{split}.tfrecords"
SPLITS = ["train", "val", "test"]
SEQUENCE_LENGTH = {"inputs": 1280, "targets1": 512, "targets2": 512}

def get_dataset(split):
    tfrecords_path = TFRECORDS_PATH.format(split=split)
    serialized_dataset = tf.data.TFRecordDataset(tfrecords_path)
    feature_description = {}
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
            SEQUENCE_LENGTH["targets1"],
            tf.int64,
            default_value=[0]*SEQUENCE_LENGTH["targets1"]
        ),
        "targets1_position": tf.io.FixedLenFeature(
            SEQUENCE_LENGTH["targets1"],
            tf.int64,
            default_value=[0]*SEQUENCE_LENGTH["targets1"]
        ),
        "targets2": tf.io.FixedLenFeature(
            SEQUENCE_LENGTH["targets2"],
            tf.int64,
            default_value=[0]*SEQUENCE_LENGTH["targets2"]
        ),
        "targets2_position": tf.io.FixedLenFeature(
            SEQUENCE_LENGTH["targets2"],
            tf.int64,
            default_value=[0]*SEQUENCE_LENGTH["targets2"]
        ),
        "inputs_segmentation": tf.io.FixedLenFeature(
            SEQUENCE_LENGTH["inputs"],
            tf.int64,
            default_value=[0]*SEQUENCE_LENGTH["inputs"]
        ),
        "targets1_segmentation": tf.io.FixedLenFeature(
            SEQUENCE_LENGTH["targets1"],
            tf.int64,
            default_value=[0]*SEQUENCE_LENGTH["targets1"]
        ),
        "targets2_segmentation": tf.io.FixedLenFeature(
            SEQUENCE_LENGTH["targets2"],
            tf.int64,
            default_value=[0]*SEQUENCE_LENGTH["targets2"]
        )
    }
    tokens_dataset = serialized_dataset.map(
        lambda x: tf.io.parse_single_example(x, feature_description)
    )
    return tokens_dataset

def serialize_tokens(*values, colnames):
    example_data = {
        colnames[i]: tf.train.Feature(
            int64_list=tf.train.Int64List(value=tokens)
        )
        for i, tokens in enumerate(values)
    }
    example = tf.train.Example(features=tf.train.Features(feature=example_data))
    return example.SerializeToString()

def tf_serialize_tokens(tf_tokens_dict):
    serialize_with_colnames = partial(
        serialize_tokens,
        colnames=list(tf_tokens_dict.keys())
    )
    tf_string = tf.py_function(
        func=serialize_with_colnames,
        inp=list(tf_tokens_dict.values()),
        Tout=tf.string
    )
    return tf.reshape(tf_string, ())

if __name__ == "__main__":
    tf.enable_eager_execution()
    for split in SPLITS:
        tsv_path = TSV_PATH.format(split=split)
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
        packed_dataset = pack_or_pad(
            dataset=tokens_dataset,
            length=SEQUENCE_LENGTH,
            pack=(split == "train"),
            ensure_eos=True
        )
        serialized_dataset = packed_dataset.map(tf_serialize_tokens)
        tfrecords_path = TFRECORDS_PATH.format(split=split)
        tfrecords_writer = tf.data.experimental.TFRecordWriter(tfrecords_path)
        tfrecords_writer.write(serialized_dataset)
        tfrecords_writer.close()