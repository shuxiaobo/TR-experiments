#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by ShaneSue on 2018/8/31
"""Module implementing the IndRNN cell"""
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.layers import base as base_layer

try:
    # TF 1.7+
    from tensorflow.python.ops.rnn_cell_impl import LayerRNNCell
except ImportError:
    from tensorflow.python.ops.rnn_cell_impl import _LayerRNNCell as LayerRNNCell
    # from tensorflow.python.ops.rnn_cell import RNNCell as LayerRNNCell


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
        self._input_kernel = self.add_variable(
            "input_kernel",
            shape = [input_depth, self._num_units],
            initializer = self._input_initializer)

        if self._recurrent_initializer is None:
            self._recurrent_initializer = init_ops.constant_initializer(1.)
        self._recurrent_kernel = self.add_variable(
            "recurrent_kernel",
            shape = [self._num_units],
            initializer = self._recurrent_initializer)

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
        output =  gate_inputs
        return output, output


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
