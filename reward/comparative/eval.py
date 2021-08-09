import os
from absl import flags

import tensorflow.compat.v1 as tf
import mesh_tensorflow

from reward.comparative.model import ComparativeRewardModel
from reward.comparative.data import SEQUENCE_LENGTH
from reward.comparative.mtf_extensions import \
    make_reward_bitransformer, _tpu_estimator_model_fn

flags.DEFINE_string(
    name="model_dir",
    default=None,
    help="Output model_dir for TPUEstimator."
)
flags.DEFINE_integer(
    name="min_checkpoint_steps",
    default=-1,
    help="Steps in checkpoint to be evaluated."
)
flags.DEFINE_string(
    name="bucket_name",
    default="seri2021-advice",
    help="Root path of a GCS bucket for data and checkpoints"
)
flags.DEFINE_integer(
    name="dataset_id",
    default=None,
    help="Dataset id (to enable evaluating different datasets)"
)
flags.DEFINE_string(
    name="split",
    default="val",
    help="Split to evaluate on."
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
FLAGS = flags.FLAGS

def main(_):
    # Monkey-patch Mesh-Tensorflow model instantiation
    mesh_tensorflow.transformer.transformer.make_bitransformer = \
        make_reward_bitransformer
    # Monkey-patch Mesh-Tensorflow TPUEstimator creation
    mesh_tensorflow.transformer.utils.tpu_estimator_model_fn = \
        _tpu_estimator_model_fn
    # Initialize model
    model = ComparativeRewardModel(
        model_dir=FLAGS.model_dir,
        tpu=os.uname()[1],
        tpu_topology='2x2', # Must be this for validation
        model_parallelism=FLAGS.model_parallelism,
        batch_size=1, # To avoid dropping observations
        sequence_length=SEQUENCE_LENGTH,
        iterations_per_loop=FLAGS.iterations_per_loop,
    )
    model.eval(
        bucket_name=FLAGS.bucket_name,
        dataset_id=FLAGS.dataset_id,
        split=FLAGS.split,
        min_checkpoint_steps=FLAGS.min_checkpoint_steps
    )

if __name__ == "__main__":
    tf.app.run()
