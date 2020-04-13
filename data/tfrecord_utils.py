"""
Go from video files to tfrecord files!
"""

import collections
import os
import tempfile
from tempfile import TemporaryDirectory

import h5py
import tensorflow as tf
from google.cloud import storage


def _resolve_maybe_gcloud_fn(gclient, fn, tmp_storage_location):
    """
    Given a filename that might be on google cloud, download it if it's NOT
    :param fn: maybe gcloud filename
    :param tmp_storage_location: the local version to use
    :return: the filename you can use later to read from
    """
    if not fn.startswith('gs://'):
        if not os.path.exists(fn):
            raise ValueError("{} doesnt exist".format(fn))
        return fn

    bucket_name, ext = fn.split('gs://', 1)[1].split('/', 1)
    blob = gclient.get_bucket(bucket_name).blob(ext)
    blob.download_to_filename(tmp_storage_location)
    return tmp_storage_location


class S3TFRecordWriter(object):
    def __init__(self, fn):
        """
        Upload to gcloud
        :param fn:
        :param buffer_size: Trying to space out idential things here by shuffling a buffer

        p(first lasts until the end,N) = (1-pflush) ^ (N/(p*buffer_size))
        each flush event removes buffer_size*p
        If the buffer size is big enough then we have good randomness I think
        """
        self.fn = fn
        if fn.startswith('gs://'):
            self.gclient = storage.Client()
            self.storage_dir = TemporaryDirectory()
            self.writer = tf.io.TFRecordWriter(os.path.join(self.storage_dir.name, 'temp.tfrecord'))
            self.bucket_name, self.file_name = self.fn.split('gs://', 1)[1].split('/', 1)

        else:
            self.gclient = None
            self.bucket_name = None
            self.file_name = None
            self.storage_dir = None
            self.writer = tf.io.TFRecordWriter(fn)

    def write(self, x):
        self.writer.write(x)

    def close(self):
        self.writer.close()

        if self.gclient is not None:
            print("UPLOADING!!!!!", flush=True)
            bucket = self.gclient.get_bucket(self.bucket_name)
            blob = bucket.blob(self.file_name)
            blob.upload_from_filename(os.path.join(self.storage_dir.name, 'temp.tfrecord'))
            self.storage_dir.cleanup()

    def __enter__(self):
        # Called when entering "with" context.
        return self

    def __exit__(self, *_):
        # Called when exiting "with" context.
        # Upload shit
        print("CALLING CLOSE")
        self.close()


################################################################################################
###############################################################################################

def int64_list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def create_seq2seq_example(encoder, context, target, id=None, return_len=False):
    """
    Creates a seq2seq example.
    :param encoder: the encoder to use.
    :param context: The sentence that's the context
    :param target: The sentence that's the target.
    :param id: Optional, an ID for this instance. we'll store it as a BPE encoded thing because why not! lol
    :param return_len: return the length also.
    :return: a featuredict for seq2seq. If return_len is True we will also return the length
    """
    features = collections.OrderedDict()
    context = [encoder.begin_title] + encoder.encode(context) + [encoder.end_title, encoder.begin_article]
    target = encoder.encode(target) + [encoder.end_article]

    feats_len = len(context) + len(target)

    features['context'] = int64_list_feature(context)
    features['target'] = int64_list_feature(target)

    if id is not None:
        assert isinstance(id, str)
        features['id'] = int64_list_feature(encoder.encode(id))

    if return_len:
        return tf.train.Example(features=tf.train.Features(feature=features)), feats_len

    return tf.train.Example(features=tf.train.Features(feature=features))


def batch_index_iterator(len_l, batch_size, skip_end=True):
    """
    Provides indices that iterate over a list
    :param len_l: int representing size of thing that we will
        iterate over
    :param batch_size: size of each batch
    :param skip_end: if true, don't iterate over the last batch
    :return: A generator that returns (start, end) tuples
        as it goes through all batches
    """
    iterate_until = len_l
    if skip_end:
        iterate_until = (len_l // batch_size) * batch_size

    for b_start in range(0, iterate_until, batch_size):
        yield (b_start, min(b_start + batch_size, len_l))


class GCSH5Writer(object):
    def __init__(self, fn):
        self.fn = fn
        if fn.startswith('gs://'):
            self.gclient = storage.Client()
            self.storage_dir = tempfile.TemporaryDirectory()
            self.writer = h5py.File(os.path.join(self.storage_dir.name, 'temp.h5'), 'w')
            self.bucket_name, self.file_name = self.fn.split('gs://', 1)[1].split('/', 1)

        else:
            self.gclient = None
            self.bucket_name = None
            self.file_name = None
            self.storage_dir = None
            assert not os.path.exists(self.fn)
            self.writer = h5py.File(self.fn)

    def create_group(self, name, track_order=None):
        return self.writer.create_group(name, track_order=track_order)

    def close(self):
        self.writer.close()

        if self.gclient is not None:
            bucket = self.gclient.get_bucket(self.bucket_name)
            blob = bucket.blob(self.file_name)
            blob.upload_from_filename(os.path.join(self.storage_dir.name, 'temp.h5'))
            self.storage_dir.cleanup()

    def __enter__(self):
        # Called when entering "with" context.
        return self

    def __exit__(self, *_):
        # Called when exiting "with" context.
        # Upload shit
        print("CALLING CLOSE")
        self.close()
