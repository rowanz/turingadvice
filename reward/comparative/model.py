import os
from tqdm import tqdm
from copy import deepcopy

import gin
import tensorflow.compat.v1 as tf
from mesh_tensorflow.transformer import utils
import mesh_tensorflow.transformer as mtf_transformer

from reward.comparative.data import get_dataset, get_checkpoint_paths
from reward.comparative.data.tsvs_to_tfrecords import SEQUENCE_LENGTH
from reward.comparative.mtf_extensions import make_reward_bitransformer
from t5.data import get_mixture_or_task, DEFAULT_SPM_PATH
from t5.models.mtf_model import \
  MtfModel, _get_latest_checkpoint_from_dir, _operative_config_path

REDDIT_TASK_NAME = "reddit_v002"

class ComparativeRewardModel(MtfModel):
  def train(self, dataset_id, steps, init_checkpoint=None):
    """
    This method is a combination of MtfModel.train and
    mtf.transformer.utils.train_model, which MtfModel.train calls. It was
    re-written to fit our tfrecords dataset, which is already tokenized.

    Args:
    dataset_id: int
      Dataset id. See reward/comparative/ops.py
    steps: int
      Number of training steps.
    init_checkpoint: str
      Read from this checkpoint path when initializing variables.
    """
    vocabulary = get_mixture_or_task(REDDIT_TASK_NAME).get_vocabulary()
    sequence_length = deepcopy(self._sequence_length)
    sequence_length.update({"targets": sequence_length["targets"] * 2})
    estimator = self.estimator(vocabulary, init_checkpoint, sequence_length)
    def input_fn(params):
      del params
      dataset = get_dataset(
        dataset_id=dataset_id,
        split="train",
        from_local=False
      )
      dataset = dataset.repeat().batch(
          self.batch_size * (self._ensemble_inputs or 1),
          drop_remainder=True
        )
      dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
      return dataset
    estimator.train(input_fn=input_fn, max_steps=steps)

  def eval(self, dataset_id, split="val", min_checkpoint_steps=None):
    """
    Evaluate model metrics on several checkpoints
    """
    ckpt_paths = get_checkpoint_paths(self._model_dir, min_checkpoint_steps)
    vocabulary = get_mixture_or_task(REDDIT_TASK_NAME).get_vocabulary()
    sequence_length = deepcopy(self._sequence_length)
    sequence_length.update({"targets": sequence_length["targets"] * 2})
    # "I have no idea why but I think this must be needed?" - Rowan
    with gin.unlock_config():
      gin.parse_config_file(_operative_config_path(self._model_dir))
    estimator = self.estimator(vocabulary, sequence_length=sequence_length)
    # 
    # Data input function for TPUEstimator
    def _input_fn(params):
      del params
      dataset = get_dataset(
        dataset_id=dataset_id,
        split=split,
        from_local=False
      )
      dataset = dataset.batch(self.batch_size, drop_remainder=True)
      dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
      return dataset
    # Count steps in dataset
    with tf.Session() as sess:
      dataset = _input_fn(None)
      steps_in_dataset = sess.run(dataset.reduce(0, lambda x,_: x + 1))
    # Evaluate all checkpoints beyond min_checkpoint_steps
    for ckpt_path in tqdm(ckpt_paths):
      metrics = estimator.evaluate(
        input_fn=_input_fn,
        steps=steps_in_dataset,
        checkpoint_path=ckpt_path,
        name=split
      )
      print(f"Metrics for ckpt '{ckpt_path}': {metrics}")

  def finetune(
    self, dataset_id, finetune_steps, pretrained_model_dir,
    tokens_per_microbatch_per_replica=None,
    pretrained_checkpoint_step=-1
    ):
    if pretrained_checkpoint_step == -1:
      checkpoint_step = _get_latest_checkpoint_from_dir(pretrained_model_dir)
    else:
      checkpoint_step = pretrained_checkpoint_step
    with gin.unlock_config():
      gin.parse_config_file(_operative_config_path(pretrained_model_dir))
      gin.bind_parameter(
        "serialize_num_microbatches.tokens_per_microbatch_per_replica",
        tokens_per_microbatch_per_replica
      )
      # gin.bind_parameter("tpu_estimator_model_fn.tpu_summaries", True)
    model_ckpt = "model.ckpt-" + str(checkpoint_step)
    self.train(
      dataset_id=dataset_id,
      steps=checkpoint_step + finetune_steps,
      init_checkpoint=os.path.join(pretrained_model_dir, model_ckpt)
    )
  
  def predict(self, input_file, output_file, checkpoint_steps=-1):
    raise NotImplementedError

  def predict_one(
    self, question, checkpoint_steps=-1
    ):
    """
    Estimate reward for a single question.
    Args:
    question: dict
      A dictionary with keys "subreddit", "date", "title", "selftext",
      "created_utc"
    checkpoint_steps: int
      Use model at this checkpoint.
    """
    mtf_transformer.make_bitransformer = make_reward_bitransformer
    if checkpoint_steps == -1:
      checkpoint_steps = _get_latest_checkpoint_from_dir(self._model_dir)
    with gin.unlock_config():
      gin.parse_config_file(_operative_config_path(self._model_dir))
    vocabulary = get_mixture_or_task(REDDIT_TASK_NAME).get_vocabulary()
    str_inputs = utils.get_inputs_from_file(input_file)
    

  def export(
    self, export_dir=None, checkpoint_step=-1,
    sentencepiece_model_path=DEFAULT_SPM_PATH
    ):
    raise NotImplementedError

  def estimator(self, vocabulary, init_checkpoint=None, sequence_length=None):
    """
    A version of MtfModel.estimator which also accepts the `sequence_length`
    parameter.
    """
    return utils.get_estimator(
        model_type=self._model_type,
        input_vocab_size=utils.inputs_vocabulary(vocabulary).vocab_size,
        output_vocab_size=utils.targets_vocabulary(vocabulary).vocab_size,
        layout_rules=self._layout_rules,
        mesh_shape=self._mesh_shape,
        model_dir=self._model_dir,
        batch_size=self.batch_size,
        sequence_length=sequence_length or self._sequence_length,
        autostack=self._autostack,
        learning_rate_schedule=self._learning_rate_schedule,
        keep_checkpoint_max=self._keep_checkpoint_max,
        save_checkpoints_steps=self._save_checkpoints_steps,
        optimizer=self._optimizer,
        predict_fn=self._predict_fn,
        variable_filter=self._variable_filter,
        ensemble_inputs=self._ensemble_inputs,
        use_tpu=self._tpu,
        tpu_job_name=self._tpu_job_name,
        iterations_per_loop=self._iterations_per_loop,
        cluster=self._cluster,
        init_checkpoint=init_checkpoint)
