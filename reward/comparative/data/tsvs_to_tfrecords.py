import os
from functools import partial

import tensorflow.compat.v1 as tf
from mesh_tensorflow.transformer.dataset import pack_or_pad

from data.to_tfrecord_t5 import encoder as TOKENIZER
from reward.comparative.data.jsonl_to_tsvs import TSV_PATH, TSV_COLNAMES
from t5.data.utils import encode_string_features

TFRECORDS_PATH = os.path.dirname(__file__) + "/{split}.tfrecords"
SPLITS = ["train", "val", "test"]
SEQUENCE_LENGTH = {"inputs": 1280, "targets": 512}

def _serialize_tokens(*values, colnames):
    example_data = {
        colnames[i]: tf.train.Feature(
            int64_list=tf.train.Int64List(value=tokens)
        )
        for i, tokens in enumerate(values)
    }
    example = tf.train.Example(features=tf.train.Features(feature=example_data))
    return example.SerializeToString()

def _tf_serialize_tokens(tf_tokens_dict):
    serialize_with_colnames = partial(
        _serialize_tokens,
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
        SEQUENCE_LENGTH.update({
            "targets1": SEQUENCE_LENGTH["targets"],
            "targets2": SEQUENCE_LENGTH["targets"]
        })
        packed_dataset = pack_or_pad(
            dataset=tokens_dataset,
            length=SEQUENCE_LENGTH,
            pack=False, # Packed training not supported
            ensure_eos=True
        )
        serialized_dataset = packed_dataset.map(_tf_serialize_tokens)
        tfrecords_path = TFRECORDS_PATH.format(split=split)
        # Shuffle dataset
        shuffled_dataset = serialized_dataset.shuffle(
            buffer_size = 10000,
            seed = 41
        )
        # Write tfrecords dataset to disk
        print(f"Writting '{split}' tfrecords")
        tfrecords_writer = tf.data.experimental.TFRecordWriter(tfrecords_path)
        tfrecords_writer.write(shuffled_dataset)