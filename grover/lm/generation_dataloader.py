"""Data loader and processing.
"""
import tensorflow as tf

from grover.lm.utils import pad_to_fixed_size, get_shape_list

slim_example_decoder = tf.contrib.slim.tfexample_decoder

###################################
# Data loading stuff v2
keys_to_features = {
    'context': tf.io.VarLenFeature(tf.int64),
    'target': tf.io.VarLenFeature(tf.int64),
    'id': tf.io.VarLenFeature(tf.int64)
}
items_to_handlers = {k: (slim_example_decoder.Tensor(k)) for k in keys_to_features.keys()}


def _decode_record(record):
    """Decodes serialized tensorflow example and returns a tensor dictionary. See keys_to_features for arguments
    """
    serialized_example = tf.reshape(record, shape=[])
    decoder = slim_example_decoder.TFExampleDecoder(keys_to_features,
                                                    items_to_handlers)
    keys = sorted(decoder.list_items())

    tensors = decoder.decode(serialized_example, items=keys)
    tensor_dict = dict(zip(keys, tensors))
    return tensor_dict


def _dataset_parser(value, max_seq_length=1025):
    """

    :param value: TFRecord to decode
    :param config: NeatConfig > model and NeatConfig > data.
    :param is_training:
    :return:
    """
    with tf.name_scope('parser'):
        data = _decode_record(value)
        data = {k:tf.cast(v, dtype=tf.int32) for k, v in data.items()}

        # 0 is padding
        seq = pad_to_fixed_size(tf.concat([
            data['context'],
            data['target'],
        ], 0), 0, [max_seq_length])

        is_target = pad_to_fixed_size(tf.concat([
            tf.fill([get_shape_list(data['context'])[0]], 0),
            tf.fill([get_shape_list(data['target'])[0]], 1),
        ], 0), 0, [max_seq_length])

        feats = {'input_ids': seq, 'is_target': is_target}
        if 'id' in data:
            feats['id'] = pad_to_fixed_size(data['id'], pad_value=0, output_shape=[16])  # Max seq lenght is 16.

    return feats


def input_fn_builder(input_files,
                     seq_length,
                     is_training,
                     num_cpu_threads=4):
    """
    :param config: NeatConfig object containing model/data

    :param is_training:
    :return:
    """

    def input_fn(params):

        # this is a reserved term
        batch_size = params['batch_size']

        # For training, we want a lot of parallel reading and shuffling.
        # For eval, we want no shuffling and parallel reading doesn't matter.
        if is_training:
            d = tf.data.Dataset.from_tensor_slices(tf.constant(input_files))
            d = d.repeat()
            d = d.shuffle(buffer_size=len(input_files))

            # `cycle_length` is the number of parallel files that get read.
            cycle_length = min(num_cpu_threads, len(input_files))

            # `sloppy` mode means that the interleaving is not exact. This adds
            # even more randomness to the training pipeline.
            d = d.apply(
                tf.data.experimental.parallel_interleave(
                    tf.data.TFRecordDataset,
                    sloppy=is_training,
                    cycle_length=cycle_length))
            d = d.shuffle(buffer_size=100)
        else:
            d = tf.data.TFRecordDataset(input_files)

        # We must `drop_remainder` on training because the TPU requires fixed
        # size dimensions. For eval, we assume we are evaluating on the CPU or GPU
        # and we *don't* want to drop the remainder, otherwise we wont cover
        # every sample.
        d = d.apply(
            tf.data.experimental.map_and_batch(
                lambda record: _dataset_parser(record, max_seq_length=seq_length),
                batch_size=batch_size,
                num_parallel_batches=num_cpu_threads,
                drop_remainder=True))
        return d

    return input_fn
