import gin
import tensorflow.compat.v1 as tf
import mesh_tensorflow as mtf
from mesh_tensorflow.transformer.transformer import \
    Unitransformer, Bitransformer, get_vocab_embedding_cls, make_layer_stack, \
    Context

from reward.comparative.data import SEQUENCE_LENGTH
from reward.comparative.loss_fn import comparative_paired_rewards_loss

def get_dims_by_name(tensor, dim_name):
    return [d for d in tensor.shape.dims if d.name == dim_name]

class ScalarOutputUnitransformer(Unitransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.output_vocab_size = None
        self.autoregressive = False
        self._reward_dim = mtf.Dimension("reward", 1)

    def _call_internal(self, context, inputs, targets=None):
        """
        Overrrides Unitransformer._call_internal
        Adds losses to context!
        """
        # del targets
        if context.mode in [tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.EVAL]:
            # Input sequences (answers) should come stacked in pairs
            assert context.length_dim.size == SEQUENCE_LENGTH["targets"] * 2
            def _get_target_pair_element(x, i):
                return mtf.slice(
                    x,
                    begin=SEQUENCE_LENGTH["targets"] * i,
                    size=SEQUENCE_LENGTH["targets"],
                    slice_dim_name="length",
                    name="split_answer_pair"
                )
            # Slice answer pairs and solve reward for each separately
            reward_pairs = []
            inputs_0 = _get_target_pair_element(inputs, 0)
            context_0 = Context(
                mode=tf.estimator.ModeKeys.PREDICT,
                losses=None,
                length_dim=get_dims_by_name(inputs_0, "length")[0],
                sequence_id=_get_target_pair_element(context.sequence_id, 0),
                subsequence_id=_get_target_pair_element(context.sequence_id, 0),
                position=_get_target_pair_element(context.position, 0),
                # Copy all other context attributes
                model=context.model,
                mesh=context.mesh,
                batch_dims=context.batch_dims,
                variable_dtype=context.variable_dtype,
                beam_dim=context.beam_dim,
                position_is_default=context.position_is_default,
                states=context.states,
                new_states=context.new_states,
                initial_position=context.initial_position,
                layer_outputs=context.layer_outputs,
                encoder_output=context.encoder_output,
                encoder_sequence_id=context.encoder_sequence_id,
                constant_states=context.constant_states,
                shared_params=context.shared_params,
                encoder_layer_outputs=context.encoder_layer_outputs,
                write_priority=context.write_priority,
                read_priority=context.read_priority,
                inputs=context.inputs,
                encoder_inputs=context.encoder_inputs
            )
            inputs_1 = _get_target_pair_element(inputs, 1)
            context_1 = Context(
                mode=tf.estimator.ModeKeys.PREDICT,
                losses=None,
                length_dim=get_dims_by_name(inputs_1, "length")[0],
                sequence_id=_get_target_pair_element(context.sequence_id, 1),
                subsequence_id=_get_target_pair_element(context.sequence_id, 1),
                position=_get_target_pair_element(context.position, 1),
                # Copy all other context attributes
                model=context.model,
                mesh=context.mesh,
                batch_dims=context.batch_dims,
                variable_dtype=context.variable_dtype,
                beam_dim=context.beam_dim,
                position_is_default=context.position_is_default,
                states=context.states,
                new_states=context.new_states,
                initial_position=context.initial_position,
                layer_outputs=context.layer_outputs,
                encoder_output=context.encoder_output,
                encoder_sequence_id=context.encoder_sequence_id,
                constant_states=context.constant_states,
                shared_params=context.shared_params,
                encoder_layer_outputs=context.encoder_layer_outputs,
                write_priority=context.write_priority,
                read_priority=context.read_priority,
                inputs=context.inputs,
                encoder_inputs=context.encoder_inputs
            )
            reward_pairs.append(self._call_internal(context_0, inputs_0, None))
            reward_pairs.append(self._call_internal(context_1, inputs_1, None))
            # Compute loss and return
            reward_pairs = mtf.stack(reward_pairs, dim_name="ans_pair", name="stack_reward")
            loss = comparative_paired_rewards_loss(
                reward_pairs,
                ans_pair_dim=get_dims_by_name(reward_pairs, "ans_pair")[0]
            )
            if context.losses:
                context.losses.append(loss)
            else:
                context.losses = [loss]
            return reward_pairs
        elif context.mode == tf.estimator.ModeKeys.PREDICT:
            assert context.length_dim.size == SEQUENCE_LENGTH["targets"]
            mesh = inputs.mesh
            if self.ensemble_dim and self.ensemble_dim not in inputs.shape.dims:
                # Training an ensemble where all models are trained on the same examples.
                inputs = mtf.broadcast(inputs, [self.ensemble_dim] + inputs.shape.dims)
                if targets:
                    targets = mtf.broadcast(
                        targets, [self.ensemble_dim] + targets.shape.dims)
            if "embedding" in context.shared_params:
                vocab_embedding = context.shared_params["embedding"]
            else:
                vocab_embedding = get_vocab_embedding_cls()(
                    mesh,
                    self.input_vocab_dim,
                    self.model_dim,
                    context.variable_dtype,
                    name="embedding",
                    ensemble_dim=self.ensemble_dim)
            x = vocab_embedding.ids_to_embedding(inputs)
            if self.positional_embedding:
                if "positional_embedding" in context.shared_params:
                    pos_emb_var = context.shared_params["positional_embedding"]
                else:
                    pos_emb_var = mtf.layers.embedding_weights(
                        mesh, self.max_length_dim, self.model_dim, context.variable_dtype,
                        "positional_embedding", ensemble_dim=self.ensemble_dim)
                if (context.length_dim is not None and
                    context.length_dim.size > self.max_length_dim.size):
                    message = (
                        "Length dimenison exceeds size of positional embedding table. "
                        "length_dim.size > max_length_dim.size %s vs %s."
                        % (context.length_dim, self.max_length_dim))
                    if context.position_is_default:
                        # Definitely getting overflow in this case.
                        raise ValueError(message)
                    else:
                        tf.logging.warning(
                            message +
                            " This may be OK if there are several shorter sequences packed "
                            "together.  Otherwise, the later positions will get zeros.")
                if context.position_is_default:
                    pos_emb = mtf.rename_dimension(
                        mtf.slice(pos_emb_var, 0, context.length_dim.size,
                                self.max_length_dim.name),
                        self.max_length_dim.name, context.length_dim.name)
                else:
                    pos_emb = mtf.gather(
                        pos_emb_var, context.position, self.max_length_dim,
                        output_shape=x.shape)
                x += pos_emb
            x = self.layer_stack.call(context, x)                                                                                                                                                                                                                                                                                
            rewards = mtf.layers.dense(
                x,
                new_dims=self._reward_dim,
                reduced_dims=get_dims_by_name(x, "d_model"),
                use_bias=False,
                variable_dtype=context.variable_dtype,
                name="reward_head"
            )
            # Squeeze out size 1 reward dimension
            squeezed_shape = mtf.Shape([d for d in rewards.shape.dims if d.name != "reward"])
            rewards = mtf.reshape(rewards, squeezed_shape, name="squeeze_reward_dim")
            # Keep reward only at EOS positions
            length_dim = get_dims_by_name(
                tensor=context.sequence_id,
                dim_name="length"
            )[0]
            shifted_segmentation = mtf.shift(
                context.sequence_id,
                offset=-1,
                dim=length_dim,
                wrap=False
            )
            is_eos = mtf.not_equal(context.sequence_id, shifted_segmentation)
            eos_rewards_long = mtf.cast(is_eos, dtype=rewards.dtype) * rewards
            eos_rewards = mtf.reduce_sum(
                eos_rewards_long,
                reduced_dim=length_dim
            )
            return eos_rewards
        else:
            raise ValueError(f"Unrecognized mode: {context.mode}")

@gin.configurable
def make_reward_bitransformer(
    input_vocab_size=gin.REQUIRED,
    output_vocab_size=gin.REQUIRED,
    layout=None,
    mesh_shape=None,
    encoder_name="encoder",
    decoder_name="decoder"
    ):
    with gin.config_scope("encoder"):
        encoder = Unitransformer(
            layer_stack=make_layer_stack(),
            input_vocab_size=input_vocab_size,
            output_vocab_size=None,
            autoregressive=False,
            name=encoder_name,
            layout=layout,
            mesh_shape=mesh_shape
        )
    with gin.config_scope("decoder"):
        decoder = ScalarOutputUnitransformer(
        layer_stack=make_layer_stack(),
        input_vocab_size=output_vocab_size,
        output_vocab_size=None,
        autoregressive=False,
        name=decoder_name,
        layout=layout,
        mesh_shape=mesh_shape
    )
    return Bitransformer(encoder, decoder)

def _predict_reward_fn(model, features, variable_dtype):
  position_kwargs = dict(
      encoder_sequence_id=features.get("inputs_segmentation", None),
      decoder_sequence_id=features.get("targets_segmentation",
                                            None),
      decoder_subsequence_id=features.get("targets_subsegmentation",
                                              None),
      encoder_position=features.get("inputs_position", None),
      decoder_position=features.get("targets_position", None),
  )
  return model.call_simple(
    inputs=features["inputs"],
    targets=features["targets"],
    compute_loss=False,
    mode=tf.estimator.ModeKeys.PREDICT,
    variable_dtype=variable_dtype,
    **position_kwargs
  )[0]

import re
import six
from tensorflow.python.ops import resources
from tensorflow.python.tpu import tpu_estimator
from mesh_tensorflow.transformer import transformer
from mesh_tensorflow.transformer.utils import \
    get_variable_dtype, _dynamic_text2self, serialize_num_microbatches

@gin.configurable
def _tpu_estimator_model_fn(model_type,
                           transformer_model,
                           model_dir,
                           use_tpu,
                           mesh_shape,
                           layout_rules,
                           batch_size,
                           sequence_length,
                           autostack,
                           keep_checkpoint_max,
                           save_checkpoints_steps,
                           learning_rate_schedule=None,
                           optimizer=None,
                           outer_batch_size=1,
                           tpu_summaries=False,
                           predict_fn=_predict_reward_fn,
                           variable_filter=None,
                           init_checkpoint=None,
                           ensemble_inputs=None):
  def my_model_fn(features, labels, mode, params=None, config=None):
    """Estimator model function.
    Args:
      features: dictionary where keys are strings like "inputs" and "targets"
        and the values are the actual values of "inputs". See TPUEstimator's
        docs for more information
      labels: ignored argument
      mode: a tf.estimator.ModeKeys
      params: dictionary containing the key "context"
      config: ignored argument
    Returns:
      a TPUEstimatorSpec
    """
    del labels, config
    global_step = tf.train.get_global_step()
    if use_tpu and "context" in params:
      ctx = params["context"]
      num_hosts = ctx.num_hosts
      host_placement_fn = ctx.tpu_host_placement_function
      device_list = [host_placement_fn(host_id=t) for t in range(num_hosts)]
      # TODO(ylc): Better estimation of replica cache size?
      replica_cache_size = 300 * 1000000  # 300M per replica
      # Worker 0 caches all the TPU binaries.
      worker0_mem = replica_cache_size * ctx.num_replicas
      devices_memeory_usage = [worker0_mem] + [0] * (num_hosts - 1)
      var_placer = mtf.utils.BalancedVariablePlacer(device_list,
                                                    devices_memeory_usage)
      mesh_devices = [""] * mesh_shape.size
      physical_shape = list(
          params["context"].device_assignment.topology.mesh_shape)
      logical_to_physical = mtf.simd_mesh_impl.auto_logical_to_physical_tpu(
          mesh_shape.to_integer_list, physical_shape)
      mesh_impl = mtf.simd_mesh_impl.SimdMeshImpl(
          mesh_shape, layout_rules, mesh_devices, ctx.device_assignment,
          logical_to_physical=logical_to_physical)
    else:
      var_placer = None
      mesh_devices = [""] * mesh_shape.size
      mesh_impl = mtf.placement_mesh_impl.PlacementMeshImpl(
          mesh_shape, layout_rules, mesh_devices)

    graph = mtf.Graph()
    mesh = mtf.Mesh(graph, "my_mesh", var_placer)

    mtf_features = {}
    for key, x in features.items():
      outer_batch_dim = mtf.Dimension("outer_batch", outer_batch_size)
      batch_dim = mtf.Dimension("batch", batch_size // outer_batch_size)
      # Some auxiliary features may have been generated in packing.
      # The names of these new features are of the form
      #   "<original_feature_name>_<suffix>", e.g. "inputs_segmentation".
      #   We look up the lengths based on the original feature name, without
      #   the "_<suffix>".
      feature_length = sequence_length[key.split("_")[0]]
      length_dim = mtf.Dimension("length", feature_length)
      ensemble_dims = ([mtf.Dimension("ensemble", ensemble_inputs)]
                       if ensemble_inputs else [])
      feature_shape = mtf.Shape(
          ensemble_dims + [outer_batch_dim, batch_dim, length_dim])
      x = tf.cast(features[key], tf.int32)
      x = tf.reshape(x, feature_shape.to_integer_list)
      if not use_tpu:
        tf.logging.info("feature %s : %s" % (key, x))
        x = tf.Print(
            x, [x], "import feature %s" % key, summarize=1000, first_n=10)
      mtf_features[key] = mtf.import_fully_replicated(
          mesh, x, feature_shape, name=key)
      if key == "targets":
        anon_targets = mtf.anonymize(mtf_features[key])

    if mode == tf.estimator.ModeKeys.PREDICT:
      def _feature_shape(key):
        feature_length = sequence_length[key.split("_")[0]]
        return mtf.Shape([
            mtf.Dimension("batch", batch_size),
            mtf.Dimension("length", feature_length)
        ])
      mtf_features = {
          k: mtf.reshape(v, _feature_shape(k))
          for k, v in six.iteritems(mtf_features)
      }
      inputs = mtf_features["inputs"]
      if predict_fn:
        mtf_samples = predict_fn(
            model=transformer_model,
            features=mtf_features,
            variable_dtype=get_variable_dtype())
      elif isinstance(transformer_model, transformer.Unitransformer):
        # pad so that there is enough room for the targets
        inputs = mtf.pad(
            inputs, [0, sequence_length["targets"]], length_dim.name)
        mtf_samples = transformer_model.sample_autoregressive(
            inputs, variable_dtype=get_variable_dtype(),
            remove_partial_sequences=True)
      elif isinstance(transformer_model,
                      (transformer.Bitransformer, transformer.StudentTeacher)):
        mtf_samples = transformer_model.decode(
            inputs, variable_dtype=get_variable_dtype())
      else:
        raise ValueError("unrecognized class")
      mtf_samples = mtf.anonymize(mtf_samples)
      inputs = mtf.anonymize(inputs)
      lowering = mtf.Lowering(graph, {mesh: mesh_impl}, autostack=autostack)
      inputs = lowering.export_to_tf_tensor(inputs)
      outputs = lowering.export_to_tf_tensor(mtf_samples)
      tf.logging.info(f"inputs: {inputs}\noutputs: {outputs}")
      predictions = {
          "inputs": inputs,
          "outputs": outputs}

      # When exporting a model, we need to communicate to TF-Serving that
      # master variables need to be copied to their slave slice variables.
      # Estimator uses a Scaffold's "local_init_op" for this purpose, so we
      # augment the default "local_init_op" here.
      #
      # The "ready_op" is also constructed here to ensure the variables
      # initialized by "local_init_op" are the same ones checked by "ready_op".
      #
      # WARNING: Any variables created outside of this model_fn()
      # (e.g. tpu_estimator/iterations_per_loop) will NOT be initialized nor
      # checked by these ops.
      def scaffold_fn():
        return tf.train.Scaffold(
            local_init_op=tf.group(
                tf.train.Scaffold.default_local_init_op(),
                lowering.copy_masters_to_slices(),
                name="mtf_local_init_op"),
            ready_op=tf.concat(
                [tf.report_uninitialized_variables(),
                 resources.report_uninitialized_resources()],
                axis=0,
                name="mtf_ready_op"))

      return tpu_estimator.TPUEstimatorSpec(
          mode=tf.estimator.ModeKeys.PREDICT,
          predictions=predictions,
          scaffold_fn=scaffold_fn,
          prediction_hooks=[mtf.MtfRestoreHook(lowering)])

    assert (mode == tf.estimator.ModeKeys.TRAIN or
            mode == tf.estimator.ModeKeys.EVAL)

    def logits_and_loss(mtf_features):
      """Compute logits and loss.
      Args:
        mtf_features: a dictionary
      Returns:
        logits: a mtf.Tensor
        loss: a mtf.Tensor
      """
      if model_type == "lm":
        if "inputs" in mtf_features:
          mtf_features = _dynamic_text2self(mtf_features)
        _, _, length_dim = mtf_features["targets"].shape
        inputs = mtf.shift(mtf_features["targets"], offset=1,
                           dim=length_dim, wrap=False)
      else:
        inputs = mtf_features["inputs"]

      if isinstance(transformer_model, transformer.Unitransformer):
        position_kwargs = dict(
            sequence_id=mtf_features.get("targets_segmentation", None),
            position=mtf_features.get("targets_position", None),
        )
      elif isinstance(
          transformer_model,
          transformer.Bitransformer) or model_type == "bi_student_teacher":
        position_kwargs = dict(
            encoder_sequence_id=mtf_features.get("inputs_segmentation", None),
            decoder_sequence_id=mtf_features.get("targets_segmentation",
                                                 None),
            decoder_subsequence_id=mtf_features.get("targets_subsegmentation",
                                                    None),
            encoder_position=mtf_features.get("inputs_position", None),
            decoder_position=mtf_features.get("targets_position", None),
        )
      else:
        raise ValueError("unrecognized class")

      return  transformer_model.call_simple(
          inputs=inputs,
          targets=mtf_features["targets"],
          compute_loss=True,
          mode=mode,
          variable_dtype=get_variable_dtype(),
          **position_kwargs)

    if mode == tf.estimator.ModeKeys.TRAIN:
      num_microbatches = serialize_num_microbatches(batch_dim,
                                                    sequence_length,
                                                    mesh_shape,
                                                    layout_rules)
      if num_microbatches > 1:
        def serialized_fn(mtf_features):
          return {
              "loss": (logits_and_loss(mtf_features)[1] / num_microbatches)}
        var_grads, loss_dict = mtf.serialize_training_step(
            mtf_features, serialized_fn, batch_dim, num_microbatches)
        loss = loss_dict["loss"]
      else:
        loss = logits_and_loss(mtf_features)[1]
        var_grads = mtf.gradients(
            [loss], [v.outputs[0] for v in graph.trainable_variables])

      if tpu_summaries:
        mtf.scalar_summary("loss", loss)

      if callable(learning_rate_schedule):
        # the following happens on CPU since TPU can't handle summaries.
        with mtf.utils.outside_all_rewrites():
          learning_rate = learning_rate_schedule(
              step=tf.train.get_global_step())
          tf.summary.scalar("learning_rate", learning_rate)
      else:
        learning_rate = learning_rate_schedule

      if isinstance(variable_filter, str):
        pattern = re.compile(variable_filter)
        variable_filter_fn = lambda v: pattern.search(v.name)
      elif variable_filter is None:
        variable_filter_fn = lambda v: True
      elif callable(variable_filter):
        variable_filter_fn = variable_filter
      else:
        raise ValueError(
            "variable_filter must be None, a string, or a callable function")
      trainable_vars = [
          v for v in graph.trainable_variables if variable_filter_fn(v)]
      trainable_var_grads = [
          g for g, v in zip(var_grads, graph.trainable_variables)
          if variable_filter_fn(v)]
      if len(trainable_vars) != len(graph.trainable_variables):
        tf.logging.info("Variables being trained:")
        tf.logging.info([v.name for v in trainable_vars])
        tf.logging.info("Variables not being trained:")
        tf.logging.info([v.name for v in graph.trainable_variables
                         if not variable_filter_fn(v)])

      update_ops = optimizer(learning_rate=learning_rate).apply_grads(
          trainable_var_grads, trainable_vars
      )

      lowering = mtf.Lowering(graph, {mesh: mesh_impl}, autostack=autostack)

      tf_loss = lowering.export_to_tf_tensor(loss)
      tf_loss = tf.cast(tf_loss, tf.float32)
      if not use_tpu:
        tf_loss = tf.Print(tf_loss, [tf_loss, tf.train.get_global_step()],
                           "step, tf_loss")

      tf_update_ops = [lowering.lowered_operation(op) for op in update_ops]
      tf_update_ops.append(tf.assign_add(global_step, 1))
      train_op = tf.group(tf_update_ops)

      if hasattr(transformer_model, "initialize"):
        with mtf.utils.outside_all_rewrites():
          transformer_model.initialize()

      if tpu_summaries:
        # has to be outside of
        # with mtf.utils.outside_all_rewrites()
        host_call = mtf.utils.create_host_call(model_dir)
        mtf.utils.remove_summaries()
      else:
        host_call = None

      with mtf.utils.outside_all_rewrites():

        if init_checkpoint:
          ckpt_vars = {v for v, _ in tf.train.list_variables(init_checkpoint)}
          global_vars = {v.op.name for v in tf.global_variables()}
          restore_vars = ckpt_vars.intersection(global_vars)
          tf.logging.info("Initializing variables from %s:", init_checkpoint)
          tf.logging.debug("\n".join(sorted(restore_vars)))
          tf.logging.info("Variables in %s but not in graph:", init_checkpoint)
          tf.logging.info("\n".join(sorted(ckpt_vars - global_vars)))
          tf.logging.info("Variables in graph but not in %s:", init_checkpoint)
          tf.logging.info("\n".join(sorted(global_vars - ckpt_vars)))
          tf.train.init_from_checkpoint(
              init_checkpoint, {v: v for v in restore_vars}
          )

        # Copy master variables to slices. Must be called first.
        restore_hook = mtf.MtfRestoreHook(lowering)
        saver = tf.train.Saver(
            tf.global_variables(),
            sharded=True,
            max_to_keep=keep_checkpoint_max,
            keep_checkpoint_every_n_hours=2,
            defer_build=False,
            save_relative_paths=True)
        tf.add_to_collection(tf.GraphKeys.SAVERS, saver)
        saver_listener = mtf.MtfCheckpointSaverListener(lowering)
        saver_hook = tf.train.CheckpointSaverHook(
            model_dir,
            save_steps=save_checkpoints_steps,
            saver=saver,
            listeners=[saver_listener])
        gin_config_saver_hook = gin.tf.GinConfigSaverHook(
            model_dir, summarize_config=True, include_step_in_filename=False)

        if use_tpu:
          return tpu_estimator.TPUEstimatorSpec(
              mode=tf.estimator.ModeKeys.TRAIN,
              loss=tf_loss,
              train_op=train_op,
              host_call=host_call,
              training_hooks=[
                  restore_hook,
                  saver_hook,
                  gin_config_saver_hook,
              ])
        else:
          return tf.estimator.EstimatorSpec(
              tf.estimator.ModeKeys.TRAIN,
              loss=tf_loss,
              train_op=train_op,
              training_chief_hooks=[
                  restore_hook,
                  saver_hook,
                  gin_config_saver_hook,
              ])
    elif mode == tf.estimator.ModeKeys.EVAL:
      rewards, loss = logits_and_loss(mtf_features)
      squeezed_shape = mtf.Shape([d for d in rewards.shape.dims if d.name != "outer_batch"])
      rewards = mtf.reshape(rewards, squeezed_shape, name="squeeze_outer_batch_dim")
      rewards = mtf.anonymize(rewards) # or export_to_tf_tensor won't work
      lowering = mtf.Lowering(graph, {mesh: mesh_impl}, autostack=autostack)
      tf_loss = tf.cast(lowering.export_to_tf_tensor(loss), tf.float32)
      tf_loss = tf.cast(tf_loss, tf.float32)
      tf_rewards = tf.cast(lowering.export_to_tf_tensor(rewards), tf.float32)
      # Metrics expect answer pair dim to be at the end
      assert len(tf_rewards.shape) == 2, \
        f"Rewards tensor should be 2D, but is {len(tf_rewards.shape)}D"
      assert tf_rewards.shape.as_list()[0] == 2, \
        "First dimension of rewards tensor should be for answer pairs"
      def simple_metrics(tf_rewards):
        """Validation metrics"""
        tf_diff_filter = tf.convert_to_tensor([[-1], [1]], dtype=tf_rewards.dtype)
        tf_reward_diff = tf.reduce_sum(tf_rewards * tf_diff_filter, axis=0)
        n_correct = tf.count_nonzero(tf_reward_diff > 0)
        n = tf_rewards.shape.as_list()[1]
        return {"ranking_accuracy": tf.metrics.mean(n_correct / n)}
      eval_metrics = (simple_metrics, [tf_rewards])
      with mtf.utils.outside_all_rewrites():
        restore_hook = mtf.MtfRestoreHook(lowering)
      return tpu_estimator.TPUEstimatorSpec(
          tf.estimator.ModeKeys.EVAL,
          evaluation_hooks=[restore_hook],
          loss=tf_loss,
          eval_metrics=eval_metrics)
  return my_model_fn
