import os

from absl import flags
import tensorflow.compat.v1 as tf
import mesh_tensorflow.transformer as mtf_transformer

from reward.comparative.mtf_extensions import make_reward_bitransformer
from reward.comparative.model import ComparativeRewardModel
from reward.comparative.data import SEQUENCE_LENGTH

flags.DEFINE_integer(
    name="dataset_id",
    default=None,
    help="Dataset id (to enable training with different datasets)"
)
flags.DEFINE_string(
    name="model_dir",
    default=None,
    help="Output model_dir for TPUEstimator"
)
flags.DEFINE_string(
    name="pretrained_model_dir",
    default=None,
    help="Pretrained model dir."
)
flags.DEFINE_string(
    name="model_size",
    default="small",
    help="Model size, must be in small, base, large, 3B, 11B"
)
flags.DEFINE_string(
    name="tpu_topology",
    default="2x2",
    help="https://github.com/google-research/text-to-text-transfer-transformer/issues/34"
)
flags.DEFINE_integer(
    name="save_checkpoints_steps",
    default=1000,
    help="How often to save the model checkpoint"
)
flags.DEFINE_integer(
    name="iterations_per_loop",
    default=1000,
    help="How many steps to make in each estimator call"
)
flags.DEFINE_integer(
    name="model_parallelism",
    default=1,
    help="Number of cores per model instance"
)
flags.DEFINE_integer(
    name="num_train_steps",
    default=10000,
    help="num train steps"
)
flags.DEFINE_integer(
    name="train_batch_size",
    default=None,
    help="Batch size for SGD"
)
flags.DEFINE_float(
    name="learning_rate",
    default=0.001,
    help="The initial learning rate for adafactor"
)
FLAGS = flags.FLAGS

def main(_):
    # Monkey-patch Mesh-Tensorflow model instantiation
    mtf_transformer.make_bitransformer = make_reward_bitransformer
    # Initialize model
    model = ComparativeRewardModel(
        model_dir=FLAGS.model_dir,
        tpu=os.uname()[1],
        tpu_topology=FLAGS.tpu_topology,
        model_parallelism=FLAGS.model_parallelism,
        batch_size=FLAGS.train_batch_size,
        sequence_length=SEQUENCE_LENGTH,
        learning_rate_schedule=FLAGS.learning_rate,
        save_checkpoints_steps=FLAGS.save_checkpoints_steps,
        keep_checkpoint_max=None,
        iterations_per_loop=FLAGS.iterations_per_loop
    )
    # Train
    model.finetune(
        dataset_id=FLAGS.dataset_id,
        finetune_steps=FLAGS.num_train_steps,
        pretrained_model_dir=FLAGS.pretrained_model_dir,
        pretrained_checkpoint_step=-1
    )

if __name__ == "__main__":
    tf.app.run()