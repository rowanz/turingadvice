import os
import json

from time import time
from absl import flags
import mesh_tensorflow
import tensorflow.compat.v1 as tf

from reward.comparative.mtf_extensions import make_reward_bitransformer
from reward.comparative.model import ComparativeRewardModel
from reward.comparative.data import SEQUENCE_LENGTH

OUTPUT_MODEL_DIR = "gs://seri2021-advice/turingadvice/reward/comparative/checkpoints/{model_size}/{model_id}/"
PARAMS_OUT_PATH = os.path.join(OUTPUT_MODEL_DIR, "params.json")
PRETRAINED_MODEL_DIR = "gs://seri2021-advice/turingadvice/baselines/t5/{model_size}/"

flags.DEFINE_integer(
    name="dataset_id",
    default=None,
    help="Dataset id (to enable training with different datasets)"
)
flags.DEFINE_string(
    name="model_size",
    default=None,
    help="T5 size to be finetuned. Must be in [small, base, large, 3B, 11B]"
)
flags.DEFINE_string(
    name="tpu_topology",
    default="2x2",
    help="https://github.com/google-research/text-to-text-transfer-transformer/issues/34"
)
flags.DEFINE_integer(
    name="save_checkpoints_steps",
    default=1000,
    help="How often to save a model checkpoint"
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
    help="Total number of training steps to perform"
)
flags.DEFINE_integer(
    name="train_batch_size",
    default=None,
    help="Batch size for SGD"
)
flags.DEFINE_integer(
    name="tokens_per_microbatch_per_replica",
    default=1280 * 2,
    help="How many tokens of input can each model replica handle?"
)
flags.DEFINE_float(
    name="learning_rate",
    default=0.001,
    help="The initial learning rate for adafactor"
)
FLAGS = flags.FLAGS

def main(_):
    # Monkey-patch Mesh-Tensorflow model instantiation
    mesh_tensorflow.transformer.transformer.make_bitransformer = \
        make_reward_bitransformer
    # Store training parameters
    model_id = int(time())
    dir_params = {"model_size": FLAGS.model_size, "model_id": model_id}
    params_out_path = PARAMS_OUT_PATH.format(**dir_params)
    params = {
        "model_size": FLAGS.model_size,
        "dataset_id": FLAGS.dataset_id,
        "learning_rate": FLAGS.learning_rate,
        "num_train_steps": FLAGS.num_train_steps,
        "train_batch_size": FLAGS.train_batch_size,
        "tokens_per_microbatch_per_replica": FLAGS.tokens_per_microbatch_per_replica,
        "iterations_per_loop": FLAGS.iterations_per_loop,
        "save_checkpoints_steps": FLAGS.save_checkpoints_steps
    }
    with tf.io.gfile.GFile(params_out_path, mode="w") as params_file:
        json.dump(params, params_file, indent=2)
    # Initialize model
    output_model_dir = OUTPUT_MODEL_DIR.format(**dir_params)
    pretrained_model_dir = PRETRAINED_MODEL_DIR.format(**dir_params)
    model = ComparativeRewardModel(
        model_dir=output_model_dir,
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
        pretrained_model_dir=pretrained_model_dir,
        pretrained_checkpoint_step=-1,
        tokens_per_microbatch_per_replica=FLAGS.tokens_per_microbatch_per_replica
    )

if __name__ == "__main__":
    tf.app.run()