import collections
import re

import six
import tensorflow as tf
import numpy as np
from tensorflow.python.lib.io import file_io


def _save_np(absolute_fn, array):
    if absolute_fn.startswith('gs://'):
        with file_io.FileIO(absolute_fn, 'w') as f:
            np.save(f, array)
    else:
        np.save(absolute_fn, array)

def assert_rank(tensor, expected_rank, name=None):
    """Raises an exception if the tensor rank is not of the expected rank.

    Args:
      tensor: A tf.Tensor to check the rank of.
      expected_rank: Python integer or list of integers, expected rank.
      name: Optional name of the tensor for the error message.

    Raises:
      ValueError: If the expected shape doesn't match the actual shape.
    """
    if name is None:
        name = tensor.name

    expected_rank_dict = {}
    if isinstance(expected_rank, six.integer_types):
        expected_rank_dict[expected_rank] = True
    else:
        for x in expected_rank:
            expected_rank_dict[x] = True

    actual_rank = tensor.shape.ndims
    if actual_rank not in expected_rank_dict:
        scope_name = tf.get_variable_scope().name
        raise ValueError(
            "For the tensor `%s` in scope `%s`, the actual rank "
            "`%d` (shape = %s) is not equal to the expected rank `%s`" %
            (name, scope_name, actual_rank, str(tensor.shape), str(expected_rank)))


def get_shape_list(tensor, expected_rank=None, name=None):
    """Returns a list of the shape of tensor, preferring static dimensions.

    Args:
      tensor: A tf.Tensor object to find the shape of.
      expected_rank: (optional) int. The expected rank of `tensor`. If this is
        specified and the `tensor` has a different rank, and exception will be
        thrown.
      name: Optional name of the tensor for the error message.

    Returns:
      A list of dimensions of the shape of tensor. All static dimensions will
      be returned as python integers, and dynamic dimensions will be returned
      as tf.Tensor scalars.
    """
    if name is None:
        name = tensor.name

    if expected_rank is not None:
        assert_rank(tensor, expected_rank, name)

    shape = tensor.shape.as_list()

    non_static_indexes = []
    for (index, dim) in enumerate(shape):
        if dim is None:
            non_static_indexes.append(index)

    if not non_static_indexes:
        return shape

    dyn_shape = tf.shape(tensor)
    for index in non_static_indexes:
        shape[index] = dyn_shape[index]
    return shape


def gelu(input_tensor):
    """Gaussian Error Linear Unit.

    This is a smoother version of the RELU.
    Original paper: https://arxiv.org/abs/1606.08415

    Args:
      input_tensor: float Tensor to perform activation.

    Returns:
      `input_tensor` with the GELU activation applied.
    """
    cdf = 0.5 * (1.0 + tf.erf(input_tensor / tf.sqrt(2.0)))
    return input_tensor * cdf


def layer_norm(input_tensor, name=None, epsilon=1e-5):
    """Run layer normalization on the last dimension of the tensor."""
    name2use = f'LayerNorm_{name}' if name is not None else name
    with tf.variable_scope(name2use, default_name='LayerNorm'):
        n_state = input_tensor.shape[-1].value
        g = tf.get_variable('gamma', [n_state], initializer=tf.constant_initializer(1))
        b = tf.get_variable('beta', [n_state], initializer=tf.constant_initializer(0))
        u = tf.reduce_mean(input_tensor, axis=-1, keepdims=True)
        s = tf.reduce_mean(tf.square(input_tensor - u), axis=-1, keepdims=True)
        input_tensor = (input_tensor - u) * tf.rsqrt(s + epsilon)
        input_tensor = input_tensor * g + b
    return input_tensor
    #
    #
    # return tf.contrib.layers.layer_norm(
    #     inputs=input_tensor, begin_norm_axis=-1, begin_params_axis=-1, scope=name)


def dropout(input_tensor, dropout_prob):
    """Perform dropout.

    Args:
      input_tensor: float Tensor.
      dropout_prob: Python float. The probability of dropping out a value (NOT of
        *keeping* a dimension as in `tf.nn.dropout`).

    Returns:
      A version of `input_tensor` with dropout applied.
    """
    if dropout_prob is None or dropout_prob == 0.0:
        return input_tensor
    output = tf.nn.dropout(input_tensor, rate=dropout_prob)
    return output


def get_attention_mask(nd, ns, *, dtype):
    """1's in the lower triangle, counting from the lower right corner.

    Same as tf.matrix_band_part(tf.ones([nd, ns]), -1, ns-nd), but doesn't produce garbage on TPUs.
    """
    i = tf.range(nd)[:, None]
    j = tf.range(ns)
    m = i >= j - ns + nd
    return tf.cast(m, dtype)


def get_assignment_map_from_checkpoint(tvars, init_checkpoint):
    """Compute the union of the current variables and checkpoint variables."""
    assignment_map = {}
    initialized_variable_names = {}

    name_to_variable = collections.OrderedDict()
    for var in tvars:
        name = var.name
        m = re.match("^(.*):\\d+$", name)
        if m is not None:
            name = m.group(1)
        name_to_variable[name] = var

    init_vars = tf.train.list_variables(init_checkpoint)

    assignment_map = collections.OrderedDict()
    for x in init_vars:
        (name, var) = (x[0], x[1])
        if name not in name_to_variable:
            continue
        assignment_map[name] = name
        initialized_variable_names[name] = 1
        initialized_variable_names[name + ":0"] = 1
    return (assignment_map, initialized_variable_names)


# TPU UTILS

# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
"""Functions specific to running TensorFlow on TPUs."""

def construct_scalar_host_call(metric_dict, model_dir, prefix=""):
    """Construct a host call to log scalars when training on TPU.

    Args:
      metric_dict: A dict of the tensors to be logged.
      model_dir: The location to write the summary.
      prefix: The prefix (if any) to prepend to the metric names.

    Returns:
      A tuple of (function, args_to_be_passed_to_said_function)
    """
    # type: (dict, str) -> (function, list)
    metric_names = list(metric_dict.keys())

    def host_call_fn(global_step, *args):
        """Training host call. Creates scalar summaries for training metrics.

        This function is executed on the CPU and should not directly reference
        any Tensors in the rest of the `model_fn`. To pass Tensors from the
        model to the `metric_fn`, provide as part of the `host_call`. See
        https://www.tensorflow.org/api_docs/python/tf/contrib/tpu/TPUEstimatorSpec
        for more information.

        Arguments should match the list of `Tensor` objects passed as the second
        element in the tuple passed to `host_call`.

        Args:
          global_step: `Tensor with shape `[batch]` for the global_step
          *args: Remaining tensors to log.

        Returns:
          List of summary ops to run on the CPU host.
        """
        step = global_step[0]
        with tf.contrib.summary.create_file_writer(
                logdir=model_dir, filename_suffix=".host_call").as_default():
            with tf.contrib.summary.always_record_summaries():
                for i, name in enumerate(metric_names):
                    tf.contrib.summary.scalar(prefix + name, args[i][0], step=step)

                return tf.contrib.summary.all_summary_ops()

    # To log the current learning rate, and gradient norm for Tensorboard, the
    # summary op needs to be run on the host CPU via host_call. host_call
    # expects [batch_size, ...] Tensors, thus reshape to introduce a batch
    # dimension. These Tensors are implicitly concatenated to
    # [params['batch_size']].
    global_step_tensor = tf.reshape(
        tf.compat.v1.train.get_or_create_global_step(), [1])
    other_tensors = [tf.reshape(metric_dict[key], [1]) for key in metric_names]

    return host_call_fn, [global_step_tensor] + other_tensors

def pad_to_fixed_size(data, pad_value, output_shape, axis=0,
                      truncate=True,
                      name=None):
    """
    Pads the data to be a fixed size in the dimensions specified by axis.

    :param data: n-dimensional input.
    :param pad_value: What we will pad with
    :param output_shape: The desired output shape. This has to cover everything, not just axis.
    :param truncate: If True (default), we will TRUNCATE in the dimensions specifed by axis if we're over.
    :param axis: The axes to pad in. Pass a list to pad multiple dims.
    :return:
    """
    with tf.name_scope(name, default_name='pad_to_fixed_size', values=output_shape):
        axes = [axis] if isinstance(axis, int) else axis

        # Truncate if too long.
        pad_data = tf.identity(data)
        if truncate:
            slice_obj = [slice(0, os_i if i in axes else None, None) for i, os_i in enumerate(output_shape)]
            pad_data = pad_data[tuple(slice_obj)]

        # Anything not being padded, we assume is the output shape.
        current_shape = get_shape_list(pad_data, expected_rank=len(output_shape))
        for i, os_i in enumerate(output_shape):
            if i not in axes:
                current_shape[i] = os_i

        asserts = []
        for ax in axes:
            asserts.append(
                tf.Assert(tf.less_equal(current_shape[ax], output_shape[ax]), [current_shape[ax], output_shape[ax], ax])
            )

        with tf.control_dependencies(asserts):
            for ax in axes:
                pad_length = output_shape[ax] - current_shape[ax]
                pad_shape = [pad_length if i == ax else cs_i
                             for i, cs_i in enumerate(current_shape)]

                paddings = pad_value * tf.ones(pad_shape, dtype=data.dtype)
                pad_data = tf.concat([pad_data, paddings], axis=ax)

                # Update the dimension we padded in
                current_shape[ax] = output_shape[ax]

        pad_data = tf.reshape(pad_data, output_shape)
        return pad_data