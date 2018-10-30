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
            if self.method == 'concat':
                embed = tf.concat([embed2, embed, embed3], -1)
            elif self.method == 'dot':
                embed = embed * embed2 * embed3
            elif self.method == 'plus':
                embed = (embed + embed2 + embed3) / 3
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
    """Modified from keras==2.0.6
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
            "linear_bias", [1],
            dtype = dtype,
            regularizer = regularizer,
            initializer = bias_initializer)
        subres0 = tf.tile(dot(droped_args[0], weights4arg0), [1, 1, q_maxlen])
        subres1 = tf.tile(tf.transpose(dot(droped_args[1], weights4arg1), perm = (0, 2, 1)), [1, c_maxlen, 1])
        subres2 = batch_dot(droped_args[0] * weights4mlu, tf.transpose(droped_args[1], perm = (0, 2, 1)))
        res = subres0 + subres1 + subres2
        nn_ops.bias_add(res, biases)
        return res


def mask_logits(inputs, mask, mask_value = -1e30):
    return inputs * mask + mask_value * (1 - mask)
