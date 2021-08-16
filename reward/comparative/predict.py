import os
import sys
from absl import flags

import tensorflow.compat.v1 as tf

from reward.comparative.model import ComparativeRewardModel
from reward.comparative.data import SEQUENCE_LENGTH, MODEL_DIR

def _define_flags():
    flags.DEFINE_string(
        name="input_path",
        default=None,
        help="Path to a tab-separated text file with columns [inputs, targets]"
    )
    flags.DEFINE_string(
        name="output_path",
        default=None,
        help="File to store predictions, one per line of input"
    )
    flags.DEFINE_string(
        name="bucket_name",
        default="seri2021-advice",
        help="Root path of a GCS bucket for data and checkpoints"
    )
    flags.DEFINE_string(
        name="model_size",
        default="seri2021-advice",
        help="Root path of a GCS bucket for data and checkpoints"
    )
    flags.DEFINE_string(
        name="model_id",
        default="seri2021-advice",
        help="Root path of a GCS bucket for data and checkpoints"
    )
    flags.DEFINE_integer(
        name="checkpoint_steps",
        default=-1,
        help="Steps in checkpoint to be used for prediction"
    )
    flags.DEFINE_integer(
        name="iterations_per_loop",
        default=1000,
        help="How many steps to make in each estimator call."
    )
    flags.DEFINE_integer(
        name="model_parallelism",
        default=8,
        help="Number of cores per model instance."
    )
    flags.DEFINE_integer(
        name="batch_size",
        default=1,
        help="Batch size. Spillover samples are ignored"
    )
    flags.DEFINE_integer(
        name="tokens_per_microbatch_per_replica",
        default=1280 * 2,
        help="How many tokens of input can each model replica handle?"
    )
    return flags.FLAGS

def main(_):
    FLAGS = _define_flags()
    FLAGS(sys.argv)
    # Initialize model
    model_dir = MODEL_DIR.format(
        bucket_name=FLAGS.bucket_name,
        model_size=FLAGS.model_size,
        model_id=FLAGS.model_id
    )
    model = ComparativeRewardModel(
        model_dir=model_dir,
        tpu=os.uname()[1],
        tpu_topology='2x2', # Must be this for validation
        model_parallelism=FLAGS.model_parallelism,
        batch_size=FLAGS.batch_size,
        sequence_length=SEQUENCE_LENGTH,
        iterations_per_loop=FLAGS.iterations_per_loop,
    )
    model.predict_from_file(
        input_path=FLAGS.input_path,
        output_path=FLAGS.output_path,
        checkpoint_steps=FLAGS.checkpoint_steps
    )

if __name__ == "__main__":
    tf.app.run()
