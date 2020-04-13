# Copyright 2019 The T5 Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""Main file for launching training/eval/predictions of mesh-transformer model."""

import sys

sys.path.append('../')
import os

import tensorflow.compat.v1 as tf
from absl import flags

import t5
from google.cloud import storage
import tempfile
import gin
from t5.models.mtf_model import _get_latest_checkpoint_from_dir
import functools
from mesh_tensorflow.transformer import utils
from t5.models.mesh_transformer import mesh_eval_dataset_fn
import numpy as np

flags.DEFINE_string(
    "model_dir", None, "Estimator model_dir")

flags.DEFINE_string(
    "model_size", "small", "Model size, must be in small, base, large, 3B, 11B")

flags.DEFINE_string(
    "validation_name", 'preds.h5',
    "Name to use")

flags.DEFINE_string(
    "t5_tfds_data_dir", None,
    "If set, this directory will be used to store datasets prepared by "
    "TensorFlow Datasets that are not available in the public TFDS GCS bucket. "
    "Note that this flag overrides the `tfds_data_dir` attribute of all "
    "`Task`s.")

flags.DEFINE_integer("checkpoint_steps", None,
                     "Load from this ckpt")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_integer("model_parallelism", 4,
                     "how to distribute them.")

flags.DEFINE_integer("batch_size", 8, "Batch size to use")

flags.DEFINE_float("learning_rate", 0.001, "The initial learning rate for adafactor.")


FLAGS = flags.FLAGS


def main(_):
    assert FLAGS.model_size in ["small", "base", "large", "3B", "11B"]

    if FLAGS.t5_tfds_data_dir:
        t5.data.set_tfds_data_dir_override(FLAGS.t5_tfds_data_dir)

    # # Add search path for gin files stored in package.
    # gin.add_config_file_search_path(
    #     pkg_resources.resource_filename(__name__, "gin"))
    # try:
    #     tf.io.gfile.makedirs(FLAGS.model_dir)
    #     suffix = 0
    #     command_filename = os.path.join(FLAGS.model_dir, "command")
    #     while tf.io.gfile.exists(command_filename):
    #         suffix += 1
    #         command_filename = os.path.join(
    #             FLAGS.model_dir, "command.{}".format(suffix))
    #     with tf.io.gfile.GFile(command_filename, "w") as f:
    #         f.write(" ".join(sys.argv))
    # except tf.errors.PermissionDeniedError:
    #     logging.info(
    #         "No write access to model directory. Skipping command logging.")

    # utils.parse_gin_defaults_and_flags()
    # Public GCS path for T5 pre-trained model checkpoints
    # Set parallelism and batch size to fit on v2-8 TPU (if possible).
    # Limit number of checkpoints to fit within 5GB (if possible).
    # model_parallelism, train_batch_size, keep_checkpoint_max = {
    #     "small": (1, 256, 16),
    #     "base": (2, 128, 8),
    #     "large": (8, 64, 4),
    #     "3B": (8, 16, 1),
    #     "11B": (8, 16, 1)}[FLAGS.model_size]
    # These sizes are for v3-8 tpu
    # so if I were to scale up to v3-1024 (lol) and use 11B, we get a batch size of 256

    # model_parallelism, train_batch_size, keep_checkpoint_max = {
    #     "small": (1, 64, 16),
    #     "base": (2, 32, 8),
    #     "large": (8, 16, 4),
    #     "3B": (8, 4, 1),
    #     "11B": (8, 2, 1)}[FLAGS.model_size]

    model = t5.models.MtfModel(
        model_dir=FLAGS.model_dir,
        tpu=os.uname()[1],
        tpu_topology='2x2', # Must be this for validation
        model_parallelism=FLAGS.model_parallelism,
        batch_size=FLAGS.batch_size,
        sequence_length={"inputs": 1280, "targets": 512},
        iterations_per_loop=FLAGS.iterations_per_loop,
    )

    # from mesh_tensorflow.transformer import utils
    # gin.bind_parameter('utils.run.mode', 'perplexity_eval')
    # gin.bind_parameter('preprocessors.num_parallel_calls.deterministic', True)
    # model.eval(
    #     mixture_or_task_name='reddit_v001',
    #     checkpoint_steps=-1,
    # )
    if FLAGS.checkpoint_steps is None:
        checkpoint_steps = _get_latest_checkpoint_from_dir(FLAGS.model_dir)
    else:
        checkpoint_steps = FLAGS.checkpoint_steps

    checkpoint_path = next(iter(utils.get_checkpoint_iterator(checkpoint_steps, FLAGS.model_dir)))

    vocabulary = t5.data.get_mixture_or_task(
        'reddit_v002').get_vocabulary()

    eval_dataset_fn = functools.partial(
        mesh_eval_dataset_fn, mixture_or_task_name='reddit_v002')
    # I have no idea why but I think this must be needed?
    with gin.unlock_config():
      gin.parse_config_file(os.path.join(FLAGS.model_dir, "operative_config.gin"))
    estimator = model.estimator(vocabulary)

    eval_datasets = eval_dataset_fn(
        sequence_length=model._sequence_length,
        vocabulary=vocabulary,
        dataset_split='validation',
    )
    assert len(eval_datasets) == 1
    eval_dataset = eval_datasets[0]

    def _input_fn(params, eval_dataset):
      del params
      return (eval_dataset.dataset_fn().map(lambda x: {k: v for k, v in x.items() if k in ('inputs', 'targets')}).repeat()
              .batch(FLAGS.batch_size,
                     drop_remainder=True)
              .prefetch(tf.data.experimental.AUTOTUNE))
    fuck = estimator.evaluate(
        input_fn=functools.partial(_input_fn, eval_dataset=eval_dataset),
        steps=50*8//FLAGS.batch_size,
        checkpoint_path=checkpoint_path,
        name=eval_dataset.name)
    print("Target ppl is {:.3f}".format(np.exp(-fuck['neg_log_perplexity'].mean())), flush=True)

if __name__ == "__main__":
    tf.app.run()
