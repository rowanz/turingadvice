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

sys.path.insert(0, '../')
from data.to_tfrecord_t5 import *

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

flags.DEFINE_string(
    "model_dir", "gs://adviceeval/t5/models/jan_8_2020/model=small~lr=0.001~epochs=10~bsize=128/", "Estimator model_dir")

flags.DEFINE_string(
    "model_size", "small", "Model size, must be in small, base, large, 3B, 11B")

flags.DEFINE_string(
    "validation_name", 'preds.h5',
    "Name to use")

flags.DEFINE_string(
    "t5_tfds_data_dir", "gs://adviceeval/t5/models/test_jan_7_2020/data/",
    "If set, this directory will be used to store datasets prepared by "
    "TensorFlow Datasets that are not available in the public TFDS GCS bucket. "
    "Note that this flag overrides the `tfds_data_dir` attribute of all "
    "`Task`s.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_integer("model_parallelism", 1,
                     "how to distribute them.")

flags.DEFINE_integer("batch_size", 1, "Batch size to use")

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

    checkpoint_steps = _get_latest_checkpoint_from_dir(FLAGS.model_dir)

    checkpoint_path = next(iter(utils.get_checkpoint_iterator(checkpoint_steps, FLAGS.model_dir)))

    ex = tokenize_for_t5_advice_training(encoder=encoder, subreddit='relationships',
                                                  date=datetime.utcnow(), title="My [20M] girlfriend [18F] of 3 months mom caught us talking on facetime while she was in the shower. How is this gonna affect the relationship?", selftext="Me and my girlfriend have been dating for 3 months and every thing is going great. We were talking on facetime and her mom was supposed to be away for the weekend, so things were getting a little freaky on the call. She then got in the shower and had the camera pointed at her, so that i could see her clearly in the shower while we talked. All of a sudden her mom came in and i heard her yelling so i paused myself but her camera was still on. When i wasnt so shocked i went back to hang up the call and i saw that her mom moved her phone to the sink. I texted my girlfriend after to see if she was ok and this was her response “She just took my phone and put it in the sink and she was like that guy doesn’t let u breath u guys are always talking I don’t understand why u guys ft while I shower u do not respect ur self or make him respect u” and that “And then she was like I will make ur dad speak with him”. I’ve met both her parents once before, and made sure that i left a good impression on them but now this happened. My girlfriend is from a colombian background so i don’t know how they will react to this! How is this going to affect the relationship going into the future?\n\nTLDR: Girlfriend’s mom caught us facetiming while my girlfriend was in the shower and the mom wants the dad to speak with me.", body="lol"*128)

    predict_input_path = os.path.join(FLAGS.model_dir, 'predict_input.txt')
    predict_output_path = os.path.join(FLAGS.model_dir, 'predict_output.txt')

    with tf.io.gfile.GFile(predict_input_path, 'w') as f:
        for _ in range(8):
            f.write(''.join(["Subreddit: ", ex['subreddit'], " Date: ", ex['date'], " Title: ", ex['title'], " Selftext: ", ex['selftext']]) + '\n')

    model.batch_size = 8
    model.predict(
        input_file=predict_input_path,
        output_file=predict_output_path,
        checkpoint_steps=checkpoint_steps,
        sampling_keep_top_p=0.95
    )
    # WOOHOO THIS WORKS!




if __name__ == "__main__":
    tf.app.run()
