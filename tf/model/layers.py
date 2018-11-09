#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by ShaneSue on 2018/8/31
"""Module implementing the IndRNN cell"""
import math
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import special_math_ops
from tensorflow.python.layers import base as base_layer

try:
    # TF 1.7+
    from tensorflow.python.ops.rnn_cell_impl import LayerRNNCell
except ImportError:
    from tensorflow.python.ops.rnn_cell_impl import _LayerRNNCell as LayerRNNCell
    # from tensorflow.python.ops.rnn_cell import RNNCell as LayerRNNCell

_BIAS_VARIABLE_NAME = "bias"
_WEIGHTS_VARIABLE_NAME = "kernel"


class IndRNNCell(LayerRNNCell):
    def __init__(self,
                 num_units,
                 recurrent_min_abs = 0,
                 recurrent_max_abs = None,
                 recurrent_kernel_initializer = None,
                 input_kernel_initializer = None,
                 activation = None,
                 reuse = None,
                 name = None):
        super(IndRNNCell, self).__init__(_reuse = reuse, name = name)

        # Inputs must be 2-dimensional.
        self.input_spec = base_layer.InputSpec(ndim = 2)

        self._num_units = num_units
        self._recurrent_min_abs = recurrent_min_abs
        self._recurrent_max_abs = recurrent_max_abs
        self._recurrent_initializer = recurrent_kernel_initializer
        self._input_initializer = input_kernel_initializer
        self._activation = activation or nn_ops.relu

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def build(self, inputs_shape):
        if inputs_shape[1].value is None:
            raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s"
                             % inputs_shape)

        input_depth = inputs_shape[1].value
        if self._input_initializer is None:
            self._input_initializer = init_ops.random_normal_initializer(mean = 0.0,
                                                                         stddev = 0.001)
        self._input_kernel = self.add_variable("input_kernel", shape = [input_depth, self._num_units], initializer = self._input_initializer)

        if self._recurrent_initializer is None:
            self._recurrent_initializer = init_ops.constant_initializer(1.)
        self._recurrent_kernel = self.add_variable("recurrent_kernel", shape = [self._num_units], initializer = self._recurrent_initializer)

        # Clip the absolute values of the recurrent weights to the specified minimum
        if self._recurrent_min_abs:
            abs_kernel = math_ops.abs(self._recurrent_kernel)
            min_abs_kernel = math_ops.maximum(abs_kernel, self._recurrent_min_abs)
            self._recurrent_kernel = math_ops.multiply(
                math_ops.sign(self._recurrent_kernel),
                min_abs_kernel
            )

        # Clip the absolute values of the recurrent weights to the specified maximum
        if self._recurrent_max_abs:
            self._recurrent_kernel = clip_ops.clip_by_value(self._recurrent_kernel,
                                                            -self._recurrent_max_abs,
                                                            self._recurrent_max_abs)

        self._bias = self.add_variable(
            "bias",
            shape = [self._num_units],
            initializer = init_ops.zeros_initializer(dtype = self.dtype))

        self.built = True

    def call(self, inputs, state):
        gate_inputs = math_ops.matmul(inputs, self._input_kernel)
        recurrent_update = math_ops.multiply(state, self._recurrent_kernel)
        gate_inputs = math_ops.add(gate_inputs, recurrent_update)
        gate_inputs = nn_ops.bias_add(gate_inputs, self._bias)
        output = self._activation(gate_inputs)
        return output, output


class ModifiedRNNCell(LayerRNNCell):
    def __init__(self, num_units, activation = nn_ops.relu, bias = False, reuse = None, name = None):
        super(ModifiedRNNCell, self).__init__(_reuse = reuse, name = name)
        self._num_units = num_units
        self.bias = bias
        self._activation = activation

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def build(self, inputs_shape):
        if inputs_shape[1].value is None:
            raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s"
                             % inputs_shape)
        init_ih = init_ops.identity_initializer
        self._ih = tf.get_variable(name = 'ih', shape = [inputs_shape[1], self.state_size], initializer = init_ih)

        init_hh = init_ops.identity_initializer
        self._hh = tf.get_variable(name = 'hh', shape = [self.state_size, inputs_shape[1]], initializer = init_hh)

        if self.bias:
            self._bias = self.add_variable("bias", shape = [self.state_size], initializer = init_ops.zeros_initializer)
        self.built = True

    def call(self, inputs, state):
        gate_inputs = math_ops.matmul(inputs, self._ih)
        recurrent_update = math_ops.matmul(state, math_ops.sin(math_ops.matmul(self._hh, self._ih)))
        gate_inputs = math_ops.add(gate_inputs, recurrent_update)
        if self.bias:
            gate_inputs = nn_ops.bias_add(gate_inputs, self._bias)
        output = gate_inputs
        return output, output


class ModifiedGRUCell(LayerRNNCell):

    def __init__(self,
                 num_units,
                 activation = None,
                 num_heads = 4,
                 reuse = None,
                 kernel_initializer = None,
                 bias_initializer = None,
                 name = None):
        super(ModifiedGRUCell, self).__init__(_reuse = reuse, name = name)

        # Inputs must be 2-dimensional.
        self.input_spec = base_layer.InputSpec(ndim = 2)

        self._num_units = num_units
        self._activation = activation or math_ops.tanh
        self._num_heads = num_heads
        self._kernel_initializer = kernel_initializer
        self._bias_initializer = bias_initializer
        self.filter_size = 3

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def build(self, inputs_shape):
        if inputs_shape[1].value is None:
            raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s"
                             % inputs_shape)

        input_depth = inputs_shape[1].value
        self._num_blocks = (self._num_units + input_depth) // self._num_heads
        self._input_depth = input_depth
        self._gate_kernel = self.add_variable("gates/%s" % _WEIGHTS_VARIABLE_NAME, shape = [self._num_blocks, self._num_blocks],
                                              initializer = self._kernel_initializer)

        self._gate_kernel2 = self.add_variable("gates/%s" % _BIAS_VARIABLE_NAME,
                                               shape = [self._num_units],
                                               initializer = self._kernel_initializer)

        self._tabel_kernel2 = self.add_variable('tabel/%s' % _WEIGHTS_VARIABLE_NAME, shape = [self._input_depth + self._num_units, self._num_units],
                                                initializer = self._kernel_initializer)

        self._candidate_kernel = self.add_variable("candidate/%s" % _WEIGHTS_VARIABLE_NAME,
                                                   shape = [self._num_units * 4, self._num_units], initializer = self._kernel_initializer)
        self._candidate_bias = self.add_variable("candidate/%s" % _BIAS_VARIABLE_NAME,
                                                 shape = [self._num_units],
                                                 initializer = (self._bias_initializer if self._bias_initializer is not None
                                                                else init_ops.zeros_initializer(dtype = self.dtype)))

        self.built = True

    def call(self, inputs, state):
        """Gated recurrent unit (GRU) with nunits cells."""

        # attention gate
        inputs_state_con = array_ops.concat([inputs, state], 1)
        gate_inputs = tf.reshape(inputs_state_con, [tf.shape(inputs)[0], self._num_heads, self._num_blocks])
        gate_inputs_att = self._activation(special_math_ops.einsum('bij,jk->bik', gate_inputs, self._gate_kernel))
        gate_inputs = gate_inputs_att * gate_inputs
        inputs_state_con = math_ops.matmul(tf.reshape(gate_inputs, shape = [tf.shape(inputs)[0], self._input_depth + self._num_units]), self._tabel_kernel2)

        gate2 = inputs_state_con * self._activation(self._gate_kernel2)

        state = tf.reshape(inputs_state_con, shape = [tf.shape(inputs)[0], self._num_units]) + gate2
        return state, state


class VanillaRNNCell(LayerRNNCell):
    def __init__(self, num_units, activation = nn_ops.relu, bias = False, reuse = None, name = None):
        super(VanillaRNNCell, self).__init__(_reuse = reuse, name = name)
        self._num_units = num_units
        self.bias = bias
        self._activation = activation

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def build(self, inputs_shape):
        if inputs_shape[1].value is None:
            raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s"
                             % inputs_shape)
        init_ih = init_ops.identity_initializer
        self._ih = tf.get_variable(name = 'ih', shape = [inputs_shape[1], self.state_size], initializer = init_ih)

        init_hh = init_ops.identity_initializer
        self._hh = tf.get_variable(name = 'hh', shape = [self.state_size, self.state_size], initializer = init_hh)

        if self.bias:
            self._bias = self.add_variable("bias", shape = [self.state_size], initializer = init_ops.zeros_initializer)
        self.built = True

    def call(self, inputs, state):
        gate_inputs = math_ops.matmul(inputs, self._ih)
        recurrent_update = math_ops.matmul(state, self._hh)
        gate_inputs = math_ops.add(gate_inputs, recurrent_update)
        if self.bias:
            gate_inputs = nn_ops.bias_add(gate_inputs, self._bias)
        output = self._activation(gate_inputs)
        return output, output


class ClassifiedNet():

    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, x, y):
        with tf.variable_scope(name_or_scope = 'classify', reuse = False) as sc:
            feature = tf.concat([x, y, x - y, x * y], -1)

            w = tf.get_variable('w', shape = [feature.get_shape()[-1], self.output_size], dtype = tf.float32, initializer = tf.random_uniform_initializer)
            bias = tf.get_variable('bias', shape = [self.output_size], initializer = tf.random_uniform_initializer, dtype = tf.float32)

            cls_result = tf.nn.xw_plus_b(x = feature, weights = w, biases = bias)
            # cls_result = tf.einsum('bij,jk->bik', feature, w)
        return cls_result


class ContextEmbedding:
    def __init__(self, context_size = 3, method = 'concat'):
        """
        use the word context embedding
        :param context_size:
        :param method: 'concat' , 'dot', 'matmul'
        """
        self.context_size = context_size
        self.method = method

    def __call__(self, x, embedding_matrix):
        """
        concat or dot or matmul the context words embedding.
        :param x:
        :param embedding_matrix:
        :return:
        """
        with tf.variable_scope('context_embedding', reuse = False) as scp:
            embed = tf.nn.embedding_lookup(embedding_matrix, x, max_norm = 1.)
            embed2 = tf.nn.embedding_lookup(embedding_matrix,
                                            tf.concat([tf.zeros(shape = [tf.shape(embed)[0], 1], dtype = tf.int64), x[:, 1:]], -1), max_norm = 2.)
            embed3 = tf.nn.embedding_lookup(embedding_matrix,
                                            tf.concat([x[:, :-1], tf.zeros(shape = [tf.shape(embed)[0], 1], dtype = tf.int64)], -1), max_norm = 2.)
            embed4 = tf.nn.embedding_lookup(embedding_matrix,
                                            tf.concat([tf.zeros(shape = [tf.shape(embed)[0], 2], dtype = tf.int64), x[:, 2:]], -1), max_norm = 2.)
            embed5 = tf.nn.embedding_lookup(embedding_matrix,
                                            tf.concat([x[:, :-2], tf.zeros(shape = [tf.shape(embed)[0], 2], dtype = tf.int64)], -1), max_norm = 2.)
            if self.method == 'concat':
                embed = tf.concat([embed2, embed, embed3, embed4, embed5], -1)
            elif self.method == 'dot':
                embed = embed * embed2 * embed3
            elif self.method == 'plus':
                embed = (embed + embed2 + embed3) / 3
            elif self.method == 'stack':
                embed = tf.stack([embed2, embed, embed3], 1)
            elif self.method == 'matmul':
                w = tf.get_variable('w', shape = [embed.get_shape()[-1], embed.get_shape()[-1]], initializer = init_ops.identity_initializer)
                embed = math_ops.tensordot(tf.stack([embed2, embed, embed3], axis = -2), w, axes = [[3], [0]]) / math_ops.sqrt(
                    tf.cast(embed.get_shape()[-1], tf.float32))
                embed = tf.reshape(embed, shape = [-1, embed.get_shape()[1], embed.get_shape()[-1] * 3])
            else:
                raise NotImplementedError('No such method named {}'.format(self.method))
        return embed


def tensor_layer_norm(x, state_name = 'norm'):
    EPSILON = 1e-6
    x_shape = x.get_shape()
    dims = x_shape.ndims
    params_shape = x_shape[-1:]
    if dims == 3:
        m, v = tf.nn.moments(x, [1, 2], keep_dims = True)
    elif dims == 4:
        m, v = tf.nn.moments(x, [1, 2, 3], keep_dims = True)
    elif dims == 5:
        m, v = tf.nn.moments(x, [1, 2, 3, 4], keep_dims = True)
    else:
        raise ValueError('input tensor for layer normalization must be rank 4 or 5.')
    b = tf.get_variable(state_name + 'b', initializer = tf.zeros(params_shape))
    s = tf.get_variable(state_name + 's', initializer = tf.ones(params_shape))
    x_tln = tf.nn.batch_normalization(x, m, v, b, s, EPSILON)
    return x_tln


initializer = lambda: tf.contrib.layers.variance_scaling_initializer(factor = 1.0,
                                                                     mode = 'FAN_AVG',
                                                                     uniform = True,
                                                                     dtype = tf.float32)
initializer_relu = lambda: tf.contrib.layers.variance_scaling_initializer(factor = 2.0,
                                                                          mode = 'FAN_IN',
                                                                          uniform = False,
                                                                          dtype = tf.float32)
regularizer = tf.contrib.layers.l2_regularizer(scale = 3e-7)


def ndim(x):
    """Copied from keras==2.0.6
    Returns the number of axes in a tensor, as an integer.

    # Arguments
        x: Tensor or variable.

    # Returns
        Integer (scalar), number of axes.

    # Examples
    ```python
        >>> from keras import backend as K
        >>> inputs = K.placeholder(shape=(2, 4, 5))
        >>> val = np.array([[1, 2], [3, 4]])
        >>> kvar = K.variable(value=val)
        >>> K.ndim(inputs)
        3
        >>> K.ndim(kvar)
        2
    ```
    """
    dims = x.get_shape()._dims
    if dims is not None:
        return len(dims)
    return None


def dot(x, y):
    """ matmul
    Modified from keras==2.0.6
    Multiplies 2 tensors (and/or variables) and returns a *tensor*.

    When attempting to multiply a nD tensor
    with a nD tensor, it reproduces the Theano behavior.
    (e.g. `(2, 3) * (4, 3, 5) -> (2, 4, 5)`)

    # Arguments
        x: Tensor or variable.
        y: Tensor or variable.

    # Returns
        A tensor, dot product of `x` and `y`.
    """
    if ndim(x) is not None and (ndim(x) > 2 or ndim(y) > 2):
        x_shape = []
        for i, s in zip(x.get_shape().as_list(), tf.unstack(tf.shape(x))):
            if i is not None:
                x_shape.append(i)
            else:
                x_shape.append(s)
        x_shape = tuple(x_shape)
        y_shape = []
        for i, s in zip(y.get_shape().as_list(), tf.unstack(tf.shape(y))):
            if i is not None:
                y_shape.append(i)
            else:
                y_shape.append(s)
        y_shape = tuple(y_shape)
        y_permute_dim = list(range(ndim(y)))
        y_permute_dim = [y_permute_dim.pop(-2)] + y_permute_dim
        xt = tf.reshape(x, [-1, x_shape[-1]])
        yt = tf.reshape(tf.transpose(y, perm = y_permute_dim), [y_shape[-2], -1])
        return tf.reshape(tf.matmul(xt, yt),
                          x_shape[:-1] + y_shape[:-2] + y_shape[-1:])
    if isinstance(x, tf.SparseTensor):
        out = tf.sparse_tensor_dense_matmul(x, y)
    else:
        out = tf.matmul(x, y)
    return out


def batch_dot(x, y, axes = None):
    """Copy from keras==2.0.6
    Batchwise dot product.

    `batch_dot` is used to compute dot product of `x` and `y` when
    `x` and `y` are data in batch, i.e. in a shape of
    `(batch_size, :)`.
    `batch_dot` results in a tensor or variable with less dimensions
    than the input. If the number of dimensions is reduced to 1,
    we use `expand_dims` to make sure that ndim is at least 2.

    # Arguments
        x: Keras tensor or variable with `ndim >= 2`.
        y: Keras tensor or variable with `ndim >= 2`.
        axes: list of (or single) int with target dimensions.
            The lengths of `axes[0]` and `axes[1]` should be the same.

    # Returns
        A tensor with shape equal to the concatenation of `x`'s shape
        (less the dimension that was summed over) and `y`'s shape
        (less the batch dimension and the dimension that was summed over).
        If the final rank is 1, we reshape it to `(batch_size, 1)`.
    """
    if isinstance(axes, int):
        axes = (axes, axes)
    x_ndim = ndim(x)
    y_ndim = ndim(y)
    if x_ndim > y_ndim:
        diff = x_ndim - y_ndim
        y = tf.reshape(y, tf.concat([tf.shape(y), [1] * (diff)], axis = 0))
    elif y_ndim > x_ndim:
        diff = y_ndim - x_ndim
        x = tf.reshape(x, tf.concat([tf.shape(x), [1] * (diff)], axis = 0))
    else:
        diff = 0
    if ndim(x) == 2 and ndim(y) == 2:
        if axes[0] == axes[1]:
            out = tf.reduce_sum(tf.multiply(x, y), axes[0])
        else:
            out = tf.reduce_sum(tf.multiply(tf.transpose(x, [1, 0]), y), axes[1])
    else:
        if axes is not None:
            adj_x = None if axes[0] == ndim(x) - 1 else True
            adj_y = True if axes[1] == ndim(y) - 1 else None
        else:
            adj_x = None
            adj_y = None
        out = tf.matmul(x, y, adjoint_a = adj_x, adjoint_b = adj_y)
    if diff:
        if x_ndim > y_ndim:
            idx = x_ndim + y_ndim - 3
        else:
            idx = x_ndim - 1
        out = tf.squeeze(out, list(range(idx, idx + diff)))
    if ndim(out) == 1:
        out = tf.expand_dims(out, 1)
    return out


def conv(inputs, output_size, bias = None, activation = None, kernel_size = 1, name = "conv", reuse = None):
    with tf.variable_scope(name, reuse = reuse):
        shapes = inputs.shape.as_list()
        if len(shapes) > 4:
            raise NotImplementedError
        elif len(shapes) == 4:
            filter_shape = [1, kernel_size, shapes[-1], output_size]
            bias_shape = [1, 1, 1, output_size]
            strides = [1, 1, 1, 1]
        else:
            filter_shape = [kernel_size, shapes[-1], output_size]
            bias_shape = [1, 1, output_size]
            strides = 1
        conv_func = tf.nn.conv1d if len(shapes) == 3 else tf.nn.conv2d
        kernel_ = tf.get_variable("kernel_",
                                  filter_shape,
                                  dtype = tf.float32,
                                  regularizer = regularizer,
                                  initializer = initializer_relu() if activation is not None else initializer())
        outputs = conv_func(inputs, kernel_, strides, "VALID")
        if bias:
            outputs += tf.get_variable("bias_",
                                       bias_shape,
                                       regularizer = regularizer,
                                       initializer = tf.zeros_initializer())
        if activation is not None:
            return activation(outputs)
        else:
            return outputs


def highway(x, size = None, activation = None, num_layers = 2, scope = "highway", dropout = 0.0, reuse = None):
    with tf.variable_scope(scope, reuse):
        if size is None:
            size = x.shape.as_list()[-1]
        else:
            x = conv(x, size, name = "input_projection", reuse = reuse)
        for i in range(num_layers):
            T = conv(x, size, bias = True, activation = tf.sigmoid,
                     name = "gate_%d" % i, reuse = reuse)
            H = conv(x, size, bias = True, activation = activation,
                     name = "activation_%d" % i, reuse = reuse)
            H = tf.nn.dropout(H, 1.0 - dropout)
            x = H * T + x * (1.0 - T)
        return x


def optimized_trilinear_for_attention(args, c_maxlen, q_maxlen, input_keep_prob = 1.0,
                                      scope = 'efficient_trilinear',
                                      bias_initializer = tf.zeros_initializer(),
                                      kernel_initializer = initializer()):
    assert len(args) == 2, "just use for computing attention with two input"
    arg0_shape = args[0].get_shape().as_list()
    arg1_shape = args[1].get_shape().as_list()
    if len(arg0_shape) != 3 or len(arg1_shape) != 3:
        raise ValueError("`args` must be 3 dims (batch_size, len, dimension)")
    if arg0_shape[2] != arg1_shape[2]:
        raise ValueError("the last dimension of `args` must equal")
    arg_size = arg0_shape[2]
    dtype = args[0].dtype
    droped_args = [tf.nn.dropout(arg, input_keep_prob) for arg in args]
    with tf.variable_scope(scope):
        weights4arg0 = tf.get_variable(
            "linear_kernel4arg0", [arg_size, 1],
            dtype = dtype,
            regularizer = regularizer,
            initializer = kernel_initializer)

        weights4arg1 = tf.get_variable(
            "linear_kernel4arg1", [arg_size, 1],
            dtype = dtype,
            regularizer = regularizer,
            initializer = kernel_initializer)

        weights4mlu = tf.get_variable(
            "linear_kernel4mul", [1, 1, arg_size],
            dtype = dtype,
            regularizer = regularizer,
            initializer = kernel_initializer)

        biases = tf.get_variable(
            "linear_bias", [q_maxlen],
            dtype = dtype,
            regularizer = regularizer,
            initializer = bias_initializer)
        subres0 = tf.tile(dot(droped_args[0], weights4arg0), [1, 1, q_maxlen])  # batch * c_maxlen * q_maxlen
        subres1 = tf.tile(tf.transpose(dot(droped_args[1], weights4arg1), perm = (0, 2, 1)), [1, c_maxlen, 1])  # batch * c_maxlen * q_maxlen
        subres2 = batch_dot(droped_args[0] * weights4mlu, tf.transpose(droped_args[1], perm = (0, 2, 1)))
        res = subres0 + subres1 + subres2
        nn_ops.bias_add(res, biases)
        return res


def mask_logits(inputs, mask, mask_value = -1e30):
    return inputs * mask + mask_value * (1 - mask)


def residual_block(inputs, num_blocks, num_conv_layers, kernel_size, mask = None,
                   num_filters = 128, input_projection = False, num_heads = 8,
                   seq_len = None, scope = "res_block", is_training = True,
                   reuse = None, bias = True, dropout = 0.0):
    """
    residual conv blocks. use the conv block and self attention.
    :param inputs:
    :param num_blocks:
    :param num_conv_layers:
    :param kernel_size:
    :param mask:
    :param num_filters:
    :param input_projection:
    :param num_heads:
    :param seq_len:
    :param scope:
    :param is_training:
    :param reuse:
    :param bias:
    :param dropout:
    :return:
    """
    with tf.variable_scope(scope, reuse = reuse):
        if input_projection:
            inputs = conv(inputs, num_filters, name = "input_projection", reuse = reuse)
        outputs = inputs
        sublayer = 1
        total_sublayers = (num_conv_layers + 2) * num_blocks
        for i in range(num_blocks):
            outputs = add_timing_signal_1d(outputs)
            outputs, sublayer = conv_block(outputs, num_conv_layers, kernel_size, num_filters,
                                           seq_len = seq_len, scope = "encoder_block_%d" % i, reuse = reuse, bias = bias,
                                           dropout = dropout, sublayers = (sublayer, total_sublayers))
            outputs, sublayer = self_attention_block(outputs, num_filters, seq_len, mask = mask, num_heads = num_heads,
                                                     scope = "self_attention_layers%d" % i, reuse = reuse, is_training = is_training,
                                                     bias = bias, dropout = dropout, sublayers = (sublayer, total_sublayers))
        return outputs


def add_timing_signal_1d(x, min_timescale = 1.0, max_timescale = 1.0e4):
    """Adds a bunch of sinusoids of different frequencies to a Tensor.
    Each channel of the input Tensor is incremented by a sinusoid of a different
    frequency and phase.
    This allows attention to learn to use absolute and relative positions.
    Timing signals should be added to some precursors of both the query and the
    memory inputs to attention.
    The use of relative position is possible because sin(x+y) and cos(x+y) can be
    experessed in terms of y, sin(x) and cos(x).
    In particular, we use a geometric sequence of timescales starting with
    min_timescale and ending with max_timescale.  The number of different
    timescales is equal to channels / 2. For each timescale, we
    generate the two sinusoidal signals sin(timestep/timescale) and
    cos(timestep/timescale).  All of these sinusoids are concatenated in
    the channels dimension.
    Args:
    x: a Tensor with shape [batch, length, channels]
    min_timescale: a float
    max_timescale: a float
    Returns:
    a Tensor the same shape as x.
    """
    length = tf.shape(x)[1]
    channels = tf.shape(x)[2]
    signal = get_timing_signal_1d(length, channels, min_timescale, max_timescale)
    return x + signal


def get_timing_signal_1d(length, channels, min_timescale = 1.0, max_timescale = 1.0e4):
    """Gets a bunch of sinusoids of different frequencies.
    Each channel of the input Tensor is incremented by a sinusoid of a different
    frequency and phase.
    This allows attention to learn to use absolute and relative positions.
    Timing signals should be added to some precursors of both the query and the
    memory inputs to attention.
    The use of relative position is possible because sin(x+y) and cos(x+y) can be
    experessed in terms of y, sin(x) and cos(x).
    In particular, we use a geometric sequence of timescales starting with
    min_timescale and ending with max_timescale.  The number of different
    timescales is equal to channels / 2. For each timescale, we
    generate the two sinusoidal signals sin(timestep/timescale) and
    cos(timestep/timescale).  All of these sinusoids are concatenated in
    the channels dimension.
    Args:
    length: scalar, length of timing signal sequence.
    channels: scalar, size of timing embeddings to create. The number of
        different timescales is equal to channels / 2.
    min_timescale: a float
    max_timescale: a float
    Returns:
    a Tensor of timing signals [1, length, channels]
    """
    position = tf.to_float(tf.range(length))
    num_timescales = channels // 2
    log_timescale_increment = (
            math.log(float(max_timescale) / float(min_timescale)) /
            (tf.to_float(num_timescales) - 1))
    inv_timescales = min_timescale * tf.exp(
        tf.to_float(tf.range(num_timescales)) * -log_timescale_increment)
    scaled_time = tf.expand_dims(position, 1) * tf.expand_dims(inv_timescales, 0)
    signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis = 1)
    signal = tf.pad(signal, [[0, 0], [0, tf.mod(channels, 2)]])
    signal = tf.reshape(signal, [1, length, channels])
    return signal


def glu(x):
    """Gated Linear Units from https://arxiv.org/pdf/1612.08083.pdf"""
    x, x_h = tf.split(x, 2, axis = -1)
    return tf.sigmoid(x) * x_h


def noam_norm(x, epsilon = 1.0, scope = None, reuse = None):
    """One version of layer normalization."""
    with tf.name_scope(scope, default_name = "noam_norm", values = [x]):
        shape = x.get_shape()
        ndims = len(shape)
        return tf.nn.l2_normalize(x, ndims - 1, epsilon = epsilon) * tf.sqrt(tf.to_float(shape[-1]))


def layer_norm_compute_python(x, epsilon, scale, bias):
    """Layer norm raw computation."""
    mean = tf.reduce_mean(x, axis = [-1], keep_dims = True)
    variance = tf.reduce_mean(tf.square(x - mean), axis = [-1], keep_dims = True)
    norm_x = (x - mean) * tf.rsqrt(variance + epsilon)
    return norm_x * scale + bias


def layer_norm(x, filters = None, epsilon = 1e-6, scope = None, reuse = None):
    """Layer normalize the tensor x, averaging over the last dimension."""
    if filters is None:
        filters = x.get_shape()[-1]
    with tf.variable_scope(scope, default_name = "layer_norm", values = [x], reuse = reuse):
        scale = tf.get_variable(
            "layer_norm_scale", [filters], regularizer = regularizer, initializer = tf.ones_initializer())
        bias = tf.get_variable(
            "layer_norm_bias", [filters], regularizer = regularizer, initializer = tf.zeros_initializer())
        result = layer_norm_compute_python(x, epsilon, scale, bias)
        return result


norm_fn = layer_norm  # tf.contrib.layers.layer_norm #tf.contrib.layers.layer_norm or noam_norm


def depthwise_separable_convolution(inputs, kernel_size, num_filters,
                                    scope = "depthwise_separable_convolution",
                                    bias = True, is_training = True, reuse = None):
    """
    Layer for depth-wise separable convolution,
    :param inputs: rank 4 Tensor.
    :param kernel_size: [H,W] kernel
    :param num_filters: for point-wise conv
    :param scope:
    :param bias:
    :param is_training:
    :param reuse:
    :return:
    """
    with tf.variable_scope(scope, reuse = reuse):
        shapes = inputs.shape.as_list()
        depthwise_filter = tf.get_variable("depthwise_filter",
                                           (kernel_size[0], kernel_size[1], shapes[-1], 1),
                                           dtype = tf.float32,
                                           regularizer = regularizer,
                                           initializer = initializer_relu())
        pointwise_filter = tf.get_variable("pointwise_filter",
                                           (1, 1, shapes[-1], num_filters),
                                           dtype = tf.float32,
                                           regularizer = regularizer,
                                           initializer = initializer_relu())
        outputs = tf.nn.separable_conv2d(inputs,
                                         depthwise_filter,
                                         pointwise_filter,
                                         strides = (1, 1, 1, 1),
                                         padding = "SAME")
        if bias:
            b = tf.get_variable("bias",
                                outputs.shape[-1],
                                regularizer = regularizer,
                                initializer = tf.zeros_initializer())
            outputs += b
        outputs = tf.nn.relu(outputs)
        return outputs


def layer_dropout(inputs, residual, dropout):
    """layer residual dropout, means that use the residual data to fill the dropout position data not 0."""
    pred = tf.random_uniform([]) < dropout
    return tf.cond(pred, lambda: residual, lambda: tf.nn.dropout(inputs, 1.0 - dropout) + residual)


def conv_block(inputs, num_conv_layers, kernel_size, num_filters,
               seq_len = None, scope = "conv_block", is_training = True,
               reuse = None, bias = True, dropout = 0.0, sublayers = (1, 1)):
    """
    layer norm + depth-wise separable convolution + residual dropout
    :param inputs:
    :param num_conv_layers:
    :param kernel_size:
    :param num_filters:
    :param seq_len:
    :param scope:
    :param is_training:
    :param reuse:
    :param bias:
    :param dropout:
    :param sublayers:
    :return:
    """
    with tf.variable_scope(scope, reuse = reuse):
        outputs = tf.expand_dims(inputs, 2)
        l, L = sublayers
        for i in range(num_conv_layers):
            residual = outputs
            outputs = norm_fn(outputs, scope = "layer_norm_%d" % i, reuse = reuse)
            if (i) % 2 == 0:
                outputs = tf.nn.dropout(outputs, 1.0 - dropout)
            outputs = depthwise_separable_convolution(outputs,
                                                      kernel_size = (kernel_size, 1), num_filters = num_filters,
                                                      scope = "depthwise_conv_layers_%d" % i, is_training = is_training, reuse = reuse)
            outputs = layer_dropout(outputs, residual, dropout * float(l) / L)
            l += 1
        return tf.squeeze(outputs, 2), l


def dot_product_attention(q,
                          k,
                          v,
                          bias,
                          seq_len = None,
                          mask = None,
                          is_training = True,
                          scope = None,
                          reuse = None,
                          dropout = 0.0):
    """dot-product attention. use matmul(q,k) get attention and softmax(att) * v to get attented result
    Args:
    q: a Tensor with shape [batch, heads, length_q, depth_k]
    k: a Tensor with shape [batch, heads, length_kv, depth_k]
    v: a Tensor with shape [batch, heads, length_kv, depth_v]
    bias: bias Tensor (see attention_bias())
    is_training: a bool of training
    scope: an optional string
    Returns:
    A Tensor.
    """
    with tf.variable_scope(scope, default_name = "dot_product_attention", reuse = reuse):
        # [batch, num_heads, query_length, memory_length]
        logits = tf.matmul(q, k, transpose_b = True)
        if bias:
            b = tf.get_variable("bias",
                                logits.shape[-1],
                                regularizer = regularizer,
                                initializer = tf.zeros_initializer())
            logits += b
        if mask is not None:
            shapes = [x if x != None else -1 for x in logits.shape.as_list()]
            mask = tf.reshape(mask, [shapes[0], 1, 1, shapes[-1]])
            logits = mask_logits(logits, mask)
        weights = tf.nn.softmax(logits, name = "attention_weights")
        # dropping out the attention links for each of the heads
        weights = tf.nn.dropout(weights, 1.0 - dropout)
        return tf.matmul(weights, v)


def multihead_attention(queries, units, num_heads,
                        memory = None,
                        seq_len = None,
                        scope = "Multi_Head_Attention",
                        reuse = None,
                        mask = None,
                        is_training = True,
                        bias = True,
                        dropout = 0.0):
    """
    use 2 conv layer + multi-head attention
    if memory is None, this function is same as self multi-head attention
    conv is used to transfer the tensor to same dimension
    :param queries:
    :param units: output size
    :param num_heads:
    :param memory:
    :param seq_len: useless , had been desperated , None default
    :param scope:
    :param reuse:
    :param mask:
    :param is_training:
    :param bias:
    :param dropout:
    :return:
    """
    with tf.variable_scope(scope, reuse = reuse):
        # Self attention
        if memory is None:
            memory = queries

        memory = conv(memory, 2 * units, name = "memory_projection", reuse = reuse)
        query = conv(queries, units, name = "query_projection", reuse = reuse)

        Q = split_last_dimension(query, num_heads)
        K, V = [split_last_dimension(tensor, num_heads) for tensor in tf.split(memory, 2, axis = 2)]
        key_depth_per_head = units // num_heads
        Q *= key_depth_per_head ** -0.5
        x = dot_product_attention(Q, K, V,
                                  bias = bias,
                                  seq_len = seq_len,
                                  mask = mask,
                                  is_training = is_training,
                                  scope = "dot_product_attention",
                                  reuse = reuse, dropout = dropout)
        return combine_last_two_dimensions(tf.transpose(x, [0, 2, 1, 3]))


class ContextGRUCell(LayerRNNCell):

    def __init__(self, num_units, activation = None, num_word = 5, reuse = None, kernel_initializer = None, bias_initializer = None,
                 name = None):
        super(ContextGRUCell, self).__init__(_reuse = reuse, name = name)

        # Inputs must be 2-dimensional.
        self.input_spec = base_layer.InputSpec(ndim = 2)

        self._num_units = num_units
        self._activation = activation or math_ops.tanh
        self._kernel_initializer = kernel_initializer
        self._bias_initializer = bias_initializer
        self._num_word = num_word

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def build(self, inputs_shape):
        if inputs_shape[1].value is None:
            raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s"
                             % inputs_shape)

        input_depth = inputs_shape[1].value / self._num_word
        self.input_depth = input_depth
        self._gate_kernel = self.add_variable("gates/%s" % _WEIGHTS_VARIABLE_NAME, shape = [ self._num_units, input_depth],
                                              initializer = self._kernel_initializer)
        self._gate_bias = self.add_variable(
            "gates/%s" % _BIAS_VARIABLE_NAME,
            shape = [2 * self._num_units],
            initializer = (
                self._bias_initializer
                if self._bias_initializer is not None
                else init_ops.constant_initializer(1.0, dtype = self.dtype)))
        self._candidate_kernel = self.add_variable("candidate/%s" % _WEIGHTS_VARIABLE_NAME, shape = [input_depth + self._num_units, self._num_units],
                                                   initializer = self._kernel_initializer)
        self._candidate_bias = self.add_variable(
            "candidate/%s" % _BIAS_VARIABLE_NAME,
            shape = [self._num_units],
            initializer = (
                self._bias_initializer
                if self._bias_initializer is not None
                else init_ops.zeros_initializer(dtype = self.dtype)))

        self._rebuild_state_kernel = self.add_variable("_rebuild_state_kernel", shape = [input_depth + self._num_units, self._num_units],
                                                       initializer = self._kernel_initializer)
        self.built = True

    def call(self, inputs, state):
        """Gated recurrent unit (GRU) with nunits cells."""
        inputs_tmp = tf.transpose(array_ops.split(value = inputs, num_or_size_splits = self._num_word, axis = -1), perm = [1, 0, 2])
        memory = math_ops.matmul(state, self._gate_kernel)
        new_m = multihead_attention(tf.concat([inputs_tmp, tf.expand_dims(memory, 1)], 1), self.input_depth, self._num_word,
                                    memory = None,
                                    seq_len = None,
                                    scope = "Multi_Head_Attention",
                                    reuse = tf.AUTO_REUSE,
                                    mask = None,
                                    is_training = True,
                                    bias = True,
                                    dropout = 0.0)
        new_m = highway(x = new_m, size = None, activation = math_ops.tanh, num_layers = 2, scope = "highway", dropout = 0.0, reuse = tf.AUTO_REUSE)
        new_m = tf.reduce_max(new_m, 1)
        inputs = inputs_tmp[:, math.ceil(self._num_word / 2.0)]
        new_h = self._activation(math_ops.matmul(math_ops.tanh(tf.concat([new_m, state], -1)), self._rebuild_state_kernel))
        # gate_inputs = math_ops.matmul(array_ops.concat([inputs, state], 1), self._gate_kernel)
        # gate_inputs = nn_ops.bias_add(gate_inputs, self._gate_bias)
        #
        # value = math_ops.sigmoid(gate_inputs)
        # r, u = array_ops.split(value = value, num_or_size_splits = 2, axis = 1)
        #
        # r_state = r * state
        #
        # candidate = math_ops.matmul(array_ops.concat([inputs, r_state], 1), self._candidate_kernel)
        # candidate = nn_ops.bias_add(candidate, self._candidate_bias)
        #
        # c = self._activation(candidate)
        # new_h = u * state + (1 - u) * c
        return new_h, new_h


def self_attention_block(inputs, num_filters, seq_len, mask = None, num_heads = 8,
                         scope = "self_attention_ffn", reuse = None, is_training = True,
                         bias = True, dropout = 0.0, sublayers = (1, 1)):
    """self multi-head attention + 2 layer conv"""
    with tf.variable_scope(scope, reuse = reuse):
        l, L = sublayers
        # Self attention
        outputs = norm_fn(inputs, scope = "layer_norm_1", reuse = reuse)
        outputs = tf.nn.dropout(outputs, 1.0 - dropout)
        outputs = multihead_attention(outputs, num_filters,
                                      num_heads = num_heads, seq_len = seq_len, reuse = reuse,
                                      mask = mask, is_training = is_training, bias = bias, dropout = dropout)
        residual = layer_dropout(outputs, inputs, dropout * float(l) / L)
        l += 1
        # Feed-forward
        outputs = norm_fn(residual, scope = "layer_norm_2", reuse = reuse)
        outputs = tf.nn.dropout(outputs, 1.0 - dropout)
        outputs = conv(outputs, num_filters, True, tf.nn.relu, name = "FFN_1", reuse = reuse)
        outputs = conv(outputs, num_filters, True, None, name = "FFN_2", reuse = reuse)
        outputs = layer_dropout(outputs, residual, dropout * float(l) / L)
        l += 1
        return outputs, l


def split_last_dimension(x, n):
    """Reshape x so that the last dimension becomes two dimensions.
    The first of these two dimensions is n.
    Args:
    x: a Tensor with shape [..., m]
    n: an integer.
    Returns:
    a Tensor with shape [..., n, m/n]
    """
    old_shape = x.get_shape().dims
    last = old_shape[-1]
    new_shape = old_shape[:-1] + [n] + [last // n if last else None]
    ret = tf.reshape(x, tf.concat([tf.shape(x)[:-1], [n, -1]], 0))
    ret.set_shape(new_shape)
    return tf.transpose(ret, [0, 2, 1, 3])


def combine_last_two_dimensions(x):
    """Reshape x so that the last two dimension become one.
    Args:
    x: a Tensor with shape [..., a, b]
    Returns:
    a Tensor with shape [..., ab]
    """
    old_shape = x.get_shape().dims
    a, b = old_shape[-2:]
    new_shape = old_shape[:-2] + [a * b if a and b else None]
    ret = tf.reshape(x, tf.concat([tf.shape(x)[:-2], [-1]], 0))
    ret.set_shape(new_shape)
    return ret


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


class AttentionFlowMatchLayer(object):
    """
    实现注意力流层来计算文本-问题、问题-文本的注意力
    """

    def __init__(self, hidden_size):
        self.hidden_size = hidden_size
        self.dim_size = hidden_size * 2

    """
        根据问题向量来匹配文章向量
    """

    def __call__(self, passage_encodes, question_encodes):
        with tf.variable_scope('attn-match'):
            # bidaf
            sim_matrix = tf.matmul(passage_encodes, question_encodes, transpose_b = True)
            context2question_attn = tf.matmul(tf.nn.softmax(sim_matrix, -1), question_encodes)
            b = tf.nn.softmax(tf.expand_dims(tf.reduce_max(sim_matrix, 2), 1), -1)
            question2context_attn = tf.tile(tf.matmul(b, passage_encodes),
                                            [1, tf.shape(passage_encodes)[1], 1])

            dnm_s1 = tf.expand_dims(passage_encodes, 2)
            dnm_s2 = tf.expand_dims(question_encodes, 1)

            # concat Attn
            sjt = tf.reduce_sum(dnm_s1 + dnm_s2, 3)
            ait = tf.nn.softmax(sjt, 2)
            qtc = tf.matmul(ait, question_encodes)

            # bi-linear Attn
            sjt = tf.matmul(passage_encodes, tf.transpose(question_encodes, perm = [0, 2, 1]))
            ait = tf.nn.softmax(sjt, 2)
            qtb = tf.matmul(ait, question_encodes)

            # dot Attn
            sjt = tf.reduce_sum(dnm_s1 * dnm_s2, 3)
            ait = tf.nn.softmax(sjt, 2)
            qtd = tf.matmul(ait, question_encodes)

            # minus Attn
            sjt = tf.reduce_sum(dnm_s1 - dnm_s2, 3)
            ait = tf.nn.softmax(sjt, 2)
            qtm = tf.matmul(ait, question_encodes)

            passage_outputs = tf.concat([passage_encodes, context2question_attn,
                                         passage_encodes * context2question_attn,
                                         passage_encodes * question2context_attn, qtc, qtb, qtd, qtm], -1)

        return passage_outputs, question_encodes


class SelfMatchingLayer(object):
    """
    Implements the self-matching layer.
    """

    def __init__(self, hidden_size):
        self.hidden_size = hidden_size

    def getSelfMatchingCell(self, hidden_size, in_keep_prob = 1.0):
        cell = tf.contrib.rnn.BasicLSTMCell(hidden_size, forget_bias = 1.0, state_is_tuple = True)
        cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob = in_keep_prob)
        return cell

    def __call__(self, passage_encodes, whole_passage_encodes, p_length):
        with tf.variable_scope('self-matching'):
            # 创建cell
            # whole_passage_encodes 作为整体匹配信息

            # cell_fw = SelfMatchingCell(self.hidden_size, question_encodes)
            # cell_bw = SelfMatchingCell(self.hidden_size, question_encodes)

            cell_fw = self.getSelfMatchingCell(self.hidden_size)
            cell_bw = self.getSelfMatchingCell(self.hidden_size)

            # function:

            # self.context_to_attend = whole_passage_encodes
            # fc_context = W * context_to_attend
            self.fc_context = tf.contrib.layers.fully_connected(whole_passage_encodes, num_outputs = self.hidden_size,
                                                                activation_fn = None)
            ref_vector = passage_encodes
            # 求St的tanh部分
            G = tf.tanh(self.fc_context + tf.expand_dims(
                tf.contrib.layers.fully_connected(ref_vector, num_outputs = self.hidden_size, activation_fn = None), 1))
            # tanh部分乘以bias
            logits = tf.contrib.layers.fully_connected(G, num_outputs = 1, activation_fn = None)
            # 求a
            scores = tf.nn.softmax(logits, 1)
            # 求c
            attended_context = tf.reduce_sum(whole_passage_encodes * scores, axis = 1)
            # birnn inputs
            input_encodes = tf.concat([ref_vector, attended_context], -1)
            """
            gated
            g_t = tf.sigmoid( tc.layers.fully_connected(whole_passage_encodes,num_outputs=self.hidden_size,activation_fn=None) )
            v_tP_c_t_star = tf.squeeze(tf.multiply(input_encodes , g_t))
            input_encodes = v_tP_c_t_star
            """

            outputs, state = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw,
                                                             inputs = input_encodes,
                                                             sequence_length = p_length,
                                                             dtype = tf.float32)

            match_outputs = tf.concat(outputs, 2)
            match_state = tf.concat([state, state], 1)

            # state_fw, state_bw = state
            # c_fw, h_fw = state_fw
            # c_bw, h_bw = state_bw
        return match_outputs, match_state
