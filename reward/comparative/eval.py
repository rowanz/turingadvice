import os
from absl import flags

import tensorflow.compat.v1 as tf

from reward.comparative.model import ComparativeRewardModel
from reward.comparative.data import SEQUENCE_LENGTH

flags.DEFINE_string(
    name="split",
    default="val",
    help="Split to evaluate on."
)
flags.DEFINE_string(
    name="model_dir",
    default=None,
    help="Output model_dir for TPUEstimator."
)
flags.DEFINE_string(
    name="model_size",
    default="small",
    help="Model size, must be in small, base, large, 3B, 11B."
)
flags.DEFINE_integer(
    name="checkpoint_steps",
    default=None,
    help="Steps in checkpoint to be evaluated."
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
    default=None,
    help="Evaluation batch size."
)
FLAGS = flags.FLAGS

def main(_):
    assert FLAGS.model_size in ["small", "base", "large", "3B", "11B"]
    model = ComparativeRewardModel(
        model_dir=FLAGS.model_dir,
        tpu=os.uname()[1],
        tpu_topology='2x2', # Must be this for validation
        model_parallelism=FLAGS.model_parallelism,
        batch_size=FLAGS.batch_size,
        sequence_length=SEQUENCE_LENGTH,
        iterations_per_loop=FLAGS.iterations_per_loop,
    )
    model.eval(
        checkpoint_steps=FLAGS.checkpoint_steps,
        summary_dir=None, # Use model_dir
        split=FLAGS.split
    )

if __name__ == "__main__":
    tf.app.run()