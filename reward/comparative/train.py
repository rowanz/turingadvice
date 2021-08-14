import os
import sys
import json

from time import time
from absl import flags
import tensorflow.compat.v1 as tf

from reward.comparative.model import ComparativeRewardModel
from reward.comparative.data import SEQUENCE_LENGTH, MODEL_DIR

PARAMS_OUT_PATH = os.path.join(MODEL_DIR, "params.json")
PRETRAINED_MODEL_DIR = "gs://{bucket_name}/turingadvice/baselines/t5/{model_size}/"

def _define_flags():
    flags.DEFINE_string(
        name="model_id",
        default=None,
        help="Id of the trained model. Will use timestamp if unspecified"
    )
    flags.DEFINE_string(
        name="bucket_name",
        default="seri2021-advice",
        help="Root path of a GCS bucket for data and checkpoints"
    )
    flags.DEFINE_string(
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
    flags.DEFINE_float(
        name="dropout_rate",
        default=0.1,
        help="Dropout rate for all layers"
    )
    flags.DEFINE_boolean(
        name="freeze_encoder",
        default=False,
        help="Freeze encoder (only train decoder)"
    )
    return flags.FLAGS

def _get_variable_filter(freeze_encoder):
    if freeze_encoder:
        def variable_filter(mtf_variable):
            return "encoder" not in mtf_variable.name
    else:
        return None

def main(_):
    FLAGS = _define_flags()
    FLAGS(sys.argv)
    # Store training parameters
    model_id = FLAGS.model_id or int(time())
    dir_params = {
        "bucket_name": FLAGS.bucket_name,
        "model_size": FLAGS.model_size,
        "model_id": model_id
    }
    params_out_path = PARAMS_OUT_PATH.format(**dir_params)
    params = {
        "model_size": FLAGS.model_size,
        "dataset_id": FLAGS.dataset_id,
        "learning_rate": FLAGS.learning_rate,
        "dropout_rate": FLAGS.dropout_rate,
        "num_train_steps": FLAGS.num_train_steps,
        "train_batch_size": FLAGS.train_batch_size,
        "tokens_per_microbatch_per_replica": FLAGS.tokens_per_microbatch_per_replica,
        "iterations_per_loop": FLAGS.iterations_per_loop,
        "save_checkpoints_steps": FLAGS.save_checkpoints_steps
    }
    with tf.io.gfile.GFile(params_out_path, mode="w") as params_file:
        json.dump(params, params_file, indent=2)
    # Initialize model
    output_model_dir = MODEL_DIR.format(**dir_params)
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
        iterations_per_loop=FLAGS.iterations_per_loop,
        variable_filter=_get_variable_filter(FLAGS.freeze_encoder)
    )
    # Train
    model.finetune(
        bucket_name=FLAGS.bucket_name,
        dataset_id=FLAGS.dataset_id,
        finetune_steps=FLAGS.num_train_steps,
        pretrained_model_dir=pretrained_model_dir,
        pretrained_checkpoint_step=-1,
        dropout_rate=FLAGS.dropout_rate,
        tokens_per_microbatch_per_replica=FLAGS.tokens_per_microbatch_per_replica
    )

if __name__ == "__main__":
    tf.app.run()