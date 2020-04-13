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

flags.DEFINE_string(
    "model_dir", None, "Estimator model_dir")

flags.DEFINE_string(
    "model_size", "small", "Model size, must be in small, base, large, 3B, 11B")

flags.DEFINE_string(
    "t5_tfds_data_dir", None,
    "If set, this directory will be used to store datasets prepared by "
    "TensorFlow Datasets that are not available in the public TFDS GCS bucket. "
    "Note that this flag overrides the `tfds_data_dir` attribute of all "
    "`Task`s.")

flags.DEFINE_string(
    "tpu_topology", "2x2",
    "https://github.com/google-research/text-to-text-transfer-transformer/issues/34")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_integer("model_parallelism", 1,
                     "how to distribute them.")

flags.DEFINE_integer(
    "num_train_steps", 10000,
    "num train steps")

flags.DEFINE_integer("train_batch_size", None, "Batch size to use")

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
        tpu_topology=FLAGS.tpu_topology,
        model_parallelism=FLAGS.model_parallelism,
        batch_size=FLAGS.train_batch_size,
        sequence_length={"inputs": 1280, "targets": 512},
        learning_rate_schedule=FLAGS.learning_rate,
        save_checkpoints_steps=FLAGS.save_checkpoints_steps,
        keep_checkpoint_max=None,
        iterations_per_loop=FLAGS.iterations_per_loop,
    )

    model.finetune(
        mixture_or_task_name='reddit_v002',
        pretrained_model_dir="gs://t5-data/pretrained_models/{}".format(FLAGS.model_size),
        finetune_steps=FLAGS.num_train_steps,
    )


if __name__ == "__main__":
    tf.app.run()
