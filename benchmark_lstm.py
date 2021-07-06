# -*- coding: utf-8 -*-
#
# Copyright 2021 Amir Hadifar. All Rights Reserved.
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

import abc
import math
import time

import numpy as np
import tensorflow.compat.v1 as tf
from absl import app
from absl import logging
from sgk.sparse.ops.backend import kernels

tf.compat.v1.disable_eager_execution()
tf.disable_v2_behavior()

# benchmark params
inner_steps = 10
outer_steps = 10

# model params
batch_size = 1
seq_size = 512
hidden_size = 1024
sparsity = 0.5
block_size = 128


# class DynamicSparseGate(tf.Module):
#     def __init__(self, hz, sparsity, block_size=4):
#         assert hz % block_size == 0
#         self.sparsity = sparsity
#         self.hz = hz
#         self.block_size = block_size
#         self.n_blocks = int((hz * 4 * hz) / (block_size * block_size))
#         self.top_blocks = int(math.ceil((1. - sparsity) * self.n_blocks))
#
#         self.g_weight = tf.get_variable(initializer=tf.truncated_normal([hz, self.n_blocks], stddev=0.1),
#                                         name='gate_weight',
#                                         dtype=tf.float32)
#         self._rows = tf.Variable(
#             hz * 4,
#             trainable=False,
#             name="dg_rows",
#             dtype=tf.int32)
#         self._columns = tf.Variable(
#             hz,
#             trainable=False,
#             name="dg_columns",
#             dtype=tf.int32)
#
#     def __call__(self, inps, inp_weight):
#         gval = tf.matmul(tf.reduce_mean(inps, axis=1), self.g_weight)
#         topval = tf.math.top_k(gval, self.top_blocks)[0][:, -1]  # smallest value in topk
#         # expandval = tf.repeat(tf.cast((gval >= topval), tf.float32) * gval, self.block_size ** 2)
#         expandval = tf.cast((gval >= topval), tf.float32) * gval
#         expandval = self.block_mul(inp_weight, expandval)
#         # sparse_weight = tf.reshape(expandval, (4 * self.hz, self.hz)) * inp_weight
#         # todo: gate weight distribution
#         # return expandval
#         return self.d2s(expandval)
#
#     @tf.function
#     def block_mul(self, p, m):
#         p_x, p_y = p.shape
#         m_x, m_y = m.shape
#         m_4d = tf.reshape(m, (m_x, 1, m_y, 1))
#         m_broadcasted = tf.broadcast_to(m_4d, (m_x, p_x // m_x, m_y, p_y // m_y))
#         mp = tf.reshape(m_broadcasted, (p_x, p_y))
#         return p * mp
#
#     def d2s(self, matrix):
#         assert len(matrix.shape) == 2
#
#         # Extract the nonzero values.
#         # values = matrix.compress((matrix != 0).flatten())
#         mask = tf.math.not_equal(matrix, 0)
#         values = tf.boolean_mask(matrix, mask)
#
#         # Calculate the offset of each row.
#         mask = tf.cast(mask, tf.int32)
#         # row_offsets = np.concatenate(([0], np.cumsum(np.add.reduce(mask, axis=1))), axis=0)
#         row_offsets = tf.concat([[0], tf.cumsum(tf.reduce_sum(mask, axis=1))], axis=0)
#
#         # Create the row indices and sort them.
#         # row_indices = np.argsort(-1 * np.diff(row_offsets))
#         row_indices = tf.argsort(-(row_offsets[1:] - row_offsets[:-1]))
#
#         # Extract the column indices for the nonzero values.
#         x = mask * (tf.range(matrix.shape[1]) + 1)
#         # column_indices = x.compress((x != 0).flatten())
#         column_indices = tf.boolean_mask(x, tf.math.not_equal(matrix, 0))
#         column_indices = column_indices - 1
#
#         # Cast the desired precision.
#         values = tf.cast(values, tf.float32)
#         row_indices = tf.cast(row_indices, tf.uint32)
#         row_offsets = tf.cast(row_offsets, tf.uint32)
#         column_indices = tf.cast(column_indices, tf.uint32)
#
#         return self._rows, self._columns, values, row_indices, row_offsets, column_indices
#
#
# class BaseModel(tf.Module):
#     def __init__(self, hz, vocab_size=30000):
#         self.hidden_size = hz
#         self.vocab_size = vocab_size
#
#         # embedding
#         self.embed_weight = tf.get_variable('emb', [vocab_size, hz])
#
#         # LSTM
#         self.x_weight = tf.get_variable(initializer=tf.truncated_normal([hz * 4, hz], stddev=0.1),
#                                         name='x_weight',
#                                         dtype=tf.float32)
#         self.h_weight = tf.get_variable(initializer=tf.truncated_normal([hz * 4, hz], stddev=0.1),
#                                         name='h_weight',
#                                         dtype=tf.float32)
#
#         self.bias = tf.get_variable(initializer=tf.truncated_normal([hz * 4], stddev=0.1),
#                                     name='bias_weight',
#                                     dtype=tf.float32)
#
#         # prediction
#         self.pred_weight = tf.get_variable(initializer=tf.truncated_normal([hz, vocab_size], stddev=0.1),
#                                            name='pred_weight',
#                                            dtype=tf.float32)
#         self.pred_bias = tf.get_variable('bo', shape=[vocab_size], initializer=tf.constant_initializer(0.))
#
#         # define loss
#         self.loss_fn = tf.nn.sparse_softmax_cross_entropy_with_logits
#
#     @abc.abstractmethod
#     def lstm(self, inps):
#         pass
#
#     def pred_layer(self, inps):
#         bs, seq, hz = inps.shape
#         states_reshaped = tf.reshape(inps, [-1, hz])
#         logits = tf.matmul(states_reshaped, self.pred_weight) + self.pred_bias
#         return tf.reshape(logits, (bs, seq, -1))
#
#     def loss(self, logits, labels):
#         return tf.reduce_mean(self.loss_fn(labels=labels, logits=logits))
#
#     def __call__(self, inps):
#         inps = tf.nn.embedding_lookup(self.embed_weight, inps)
#         h = self.lstm(inps)
#         return self.pred_layer(h)
#
#
# class ToyModel(BaseModel):
#
#     @tf.function
#     def lstm(self, inps):
#         new_x_weight, new_h_weight = self.x_weight, self.h_weight  # sparsify weight
#         init_state = tf.zeros(shape=[2, inps.shape[0], self.hidden_size], dtype=tf.float32)
#
#         def step(hprev, x):
#             st_1, ct_1 = tf.unstack(hprev)
#             st_1, x = tf.transpose(st_1), tf.transpose(x)
#
#             fc_gate = tf.matmul(new_h_weight, st_1) + tf.matmul(new_x_weight, x)
#             fc_gate = tf.transpose(fc_gate) + self.bias
#             i, f, g, o = tf.split(fc_gate, 4, axis=1)
#             i, f, g, o = tf.sigmoid(i), tf.sigmoid(f), tf.tanh(g), tf.sigmoid(o)
#             ct = ct_1 * f + g * i
#             st = tf.tanh(ct) * o
#
#             return tf.stack([st, ct])
#
#         states = tf.scan(step, tf.transpose(inps, [1, 0, 2]), initializer=init_state)
#
#         return tf.transpose(states, [1, 2, 0, 3])[0]
#
#
# class ToySparseModel(BaseModel):
#     def __init__(self, hz, vocab_size=30000, block_size=4, sparsity=0.95):
#         super().__init__(hz, vocab_size)
#
#         # dynamic gate
#         self.dynamic_gate = DynamicSparseGate(hz, sparsity=sparsity, block_size=block_size)
#
#     @tf.function
#     def lstm(self, inps):
#         new_x_weight, new_h_weight, bias_weight = self.x_weight, self.h_weight, self.bias  # sparsify weight
#         rows1, columns1, values1, row_indices1, row_offsets1, column_indices1 = self.dynamic_gate(inps, new_x_weight)
#         rows2, columns2, values2, row_indices2, row_offsets2, column_indices2 = self.dynamic_gate(inps, new_h_weight)
#         # new_x_weight = self.dynamic_gate(inps, new_x_weight)
#         # new_h_weight = self.dynamic_gate(inps, new_h_weight)
#         init_state = tf.zeros(shape=[2, inps.shape[0], self.hidden_size], dtype=tf.float32)
#
#         def step(hprev, x):
#             st_1, ct_1 = tf.unstack(hprev)
#             st_1, x = tf.transpose(st_1), tf.transpose(x)
#
#             # fc_gate = tf.matmul(new_h_weight, st_1) + tf.matmul(new_x_weight, x)
#             fc_gate = \
#                 kernels.spmm(rows1, columns1, values1, row_indices1, row_offsets1, column_indices1, x, False, False) + \
#                 kernels.spmm(rows2, columns2, values2, row_indices2, row_offsets2, column_indices2, st_1, False, False)
#             fc_gate = tf.transpose(fc_gate) + bias_weight
#
#             i, f, g, o = tf.split(fc_gate, 4, axis=1)
#             i, f, g, o = tf.sigmoid(i), tf.sigmoid(f), tf.tanh(g), tf.sigmoid(o)
#             ct = ct_1 * f + g * i
#             st = tf.tanh(ct) * o
#
#             return tf.stack([st, ct])
#
#         states = tf.scan(step, tf.transpose(inps, [1, 0, 2]), initializer=init_state)
#
#         return tf.transpose(states, [1, 2, 0, 3])[0]


class DynamicSparseGate(tf.Module):
    def __init__(self, hz, sparsity, block_size=4):
        assert hz % block_size == 0
        assert (hz * 4 * hz) % (block_size * block_size) == 0

        self.sparsity = sparsity
        self.hz = hz
        self.block_size = block_size
        self.n_blocks = int((hz * 4 * hz) / (block_size * block_size))
        print('number of blocks ----------> {}'.format(self.n_blocks))
        self.top_blocks = int(math.ceil((1. - sparsity) * self.n_blocks))

        self.g_weight = tf.get_variable(initializer=tf.truncated_normal([hz, self.n_blocks], stddev=0.1),
                                        name='gate_weight',
                                        dtype=tf.float32)
        self._rows = tf.Variable(
            hz * 4,
            trainable=False,
            name="dsg_rows",
            dtype=tf.int32)
        self._columns = tf.Variable(
            hz,
            trainable=False,
            name="dsg_columns",
            dtype=tf.int32)

    def __call__(self, inps):
        gval = tf.matmul(tf.reduce_mean(inps, axis=1), self.g_weight)
        topval = tf.math.top_k(gval, self.top_blocks)[0][:, -1]  # smallest value in topk
        expandval = tf.cast((gval >= topval), tf.float32) * gval

        matrix = self.gate_loc(expandval)
        # values = tf.boolean_mask(matrix, mask)
        matrix = matrix / (tf.math.reduce_sum(matrix, axis=-1, keepdims=True) / matrix.shape[-1])
        ###############################################################
        ###############################################################
        # Extract the nonzero values.
        # values = matrix.compress((matrix != 0).flatten())
        mask = tf.math.not_equal(matrix, 0)
        # Calculate the offset of each row.
        mask = tf.cast(mask, tf.int32)
        # row_offsets = np.concatenate(([0], np.cumsum(np.add.reduce(mask, axis=1))), axis=0)
        row_offsets = tf.concat([[0], tf.cumsum(tf.reduce_sum(mask, axis=1))], axis=0)

        # Create the row indices and sort them.
        # row_indices = np.argsort(-1 * np.diff(row_offsets))
        row_indices = tf.argsort(-(row_offsets[1:] - row_offsets[:-1]))

        # Extract the column indices for the nonzero values.
        x = mask * (tf.range(matrix.shape[1]) + 1)
        # column_indices = x.compress((x != 0).flatten())
        column_indices = tf.boolean_mask(x, tf.math.not_equal(matrix, 0))
        column_indices = column_indices - 1

        # Cast the desired precision.
        # values = tf.cast(values, tf.float32)
        row_indices = tf.cast(row_indices, tf.uint32)
        row_offsets = tf.cast(row_offsets, tf.uint32)
        column_indices = tf.cast(column_indices, tf.uint32)
        return mask, self._rows, self._columns, row_indices, row_offsets, column_indices
        ###############################################################
        ###############################################################

    @tf.function
    def gate_loc(self, m):
        """p is large matrix"""
        p_x, p_y = self.hz * 4, self.hz
        m_x, m_y = m.shape
        m_4d = tf.reshape(m, (m_x, 1, m_y, 1))
        m_broadcasted = tf.broadcast_to(m_4d, (m_x, p_x // m_x, m_y, p_y // m_y))
        mp = tf.reshape(m_broadcasted, (p_x, p_y))
        return mp


class BaseModel(tf.Module):
    def __init__(self, hz, vocab_size=30000):
        self.hidden_size = hz
        self.vocab_size = vocab_size

        # embedding
        self.embed_weight = tf.get_variable('emb', [vocab_size, hz])

        # LSTM
        self.x_weight = tf.get_variable(initializer=tf.truncated_normal([hz * 4, hz], stddev=0.1),
                                        name='x_weight',
                                        dtype=tf.float32)
        self.h_weight = tf.get_variable(initializer=tf.truncated_normal([hz * 4, hz], stddev=0.1),
                                        name='h_weight',
                                        dtype=tf.float32)

        self.bias = tf.get_variable(initializer=tf.truncated_normal([hz * 4], stddev=0.1),
                                    name='bias_weight',
                                    dtype=tf.float32)

        # prediction
        self.pred_weight = tf.get_variable(initializer=tf.truncated_normal([hz, vocab_size], stddev=0.1),
                                           name='pred_weight',
                                           dtype=tf.float32)
        self.pred_bias = tf.get_variable('bo', shape=[vocab_size], initializer=tf.constant_initializer(0.))

        # define loss
        self.loss_fn = tf.nn.sparse_softmax_cross_entropy_with_logits

    @abc.abstractmethod
    def lstm(self, inps):
        pass

    def pred_layer(self, inps):
        bs, seq, hz = inps.shape
        states_reshaped = tf.reshape(inps, [-1, hz])
        logits = tf.matmul(states_reshaped, self.pred_weight) + self.pred_bias
        return tf.reshape(logits, (bs, seq, -1))

    def loss(self, logits, labels):
        return tf.reduce_mean(self.loss_fn(labels=labels, logits=logits))

    def __call__(self, inps):
        inps = tf.nn.embedding_lookup(self.embed_weight, inps)
        h = self.lstm(inps)
        return self.pred_layer(h)


class ToyModel(BaseModel):

    @tf.function
    def lstm(self, inps):
        new_x_weight, new_h_weight = self.x_weight, self.h_weight
        init_state = tf.zeros(shape=[2, inps.shape[0], self.hidden_size], dtype=tf.float32)

        def step(hprev, x):
            st_1, ct_1 = tf.unstack(hprev)
            st_1, x = tf.transpose(st_1), tf.transpose(x)

            fc_gate = tf.matmul(new_h_weight, st_1) + tf.matmul(new_x_weight, x)
            fc_gate = tf.transpose(fc_gate) + self.bias
            i, f, g, o = tf.split(fc_gate, 4, axis=1)
            i, f, g, o = tf.sigmoid(i), tf.sigmoid(f), tf.tanh(g), tf.sigmoid(o)
            ct = ct_1 * f + g * i
            st = tf.tanh(ct) * o

            return tf.stack([st, ct])

        states = tf.scan(step, tf.transpose(inps, [1, 0, 2]), initializer=init_state)

        return tf.transpose(states, [1, 2, 0, 3])[0]


class ToySparseModel(BaseModel):
    def __init__(self, hz, vocab_size=30000, block_size=4, sparsity=0.95):
        super().__init__(hz, vocab_size)

        # dynamic gate
        self.dynamic_gate = DynamicSparseGate(hz, sparsity=sparsity, block_size=block_size)

    @tf.function
    def lstm(self, inps):
        new_x_weight, new_h_weight, bias_weight = self.x_weight, self.h_weight, self.bias
        mask, rows, columns, row_indices, row_offsets, column_indices = self.dynamic_gate(inps)
        values1 = tf.boolean_mask(new_x_weight, mask)
        values2 = tf.boolean_mask(new_h_weight, mask)
        # rows2, columns2, values2, row_indices2, row_offsets2, column_indices2 = self.dynamic_gate(inps, new_h_weight)
        # new_x_weight = self.dynamic_gate(inps, new_x_weight)
        # new_h_weight = self.dynamic_gate(inps, new_h_weight)
        init_state = tf.zeros(shape=[2, inps.shape[0], self.hidden_size], dtype=tf.float32)

        def step(hprev, x):
            st_1, ct_1 = tf.unstack(hprev)
            st_1, x = tf.transpose(st_1), tf.transpose(x)

            # fc_gate = tf.matmul(new_h_weight, st_1) + tf.matmul(new_x_weight, x)
            fc_gate = \
                kernels.spmm(rows, columns, values1, row_indices, row_offsets, column_indices, x, False, False) + \
                kernels.spmm(rows, columns, values2, row_indices, row_offsets, column_indices, st_1, False, False)
            fc_gate = tf.transpose(fc_gate) + bias_weight

            i, f, g, o = tf.split(fc_gate, 4, axis=1)
            i, f, g, o = tf.sigmoid(i), tf.sigmoid(f), tf.tanh(g), tf.sigmoid(o)
            ct = ct_1 * f + g * i
            st = tf.tanh(ct) * o

            return tf.stack([st, ct])

        states = tf.scan(step, tf.transpose(inps, [1, 0, 2]), initializer=init_state)

        return tf.transpose(states, [1, 2, 0, 3])[0]


def benchmark():
    """Run repeatedly on dummy data to benchmark inference."""
    # Turn off Grappler optimizations.
    options = {"disable_meta_optimizer": True}
    tf.config.optimizer.set_experimental_options(options)

    # Run only the model body (no data pipeline) on device.
    dummy_input = tf.random.uniform([1, seq_size], minval=0, maxval=30000, dtype=tf.int32)
    # feature_shape = [batch_size, image_size, hidden_size]
    # features = {"targets": tf.zeros(feature_shape, dtype=tf.float32)}

    # model = ToySparseModel(hz=hidden_size, block_size=block_size, sparsity=sparsity)
    model = ToyModel(hz=hidden_size)

    # warm-up
    model.loss(model(dummy_input), dummy_input)

    def call_model(features):
        logits = model(features)
        return model.loss(logits, features)

    # Run the function body in a loop to amortize session overhead.
    loop_index = tf.zeros([], dtype=tf.int32)
    initial_loss = tf.zeros([])

    def loop_cond(idx, _):
        return tf.less(idx, tf.constant(inner_steps, dtype=tf.int32))

    def loop_body(idx, _):
        return idx + 1, call_model(dummy_input)

    benchmark_op = tf.while_loop(
        loop_cond,
        loop_body, [loop_index, initial_loss],
        parallel_iterations=1,
        back_prop=False)

    session_config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=False, per_process_gpu_memory_fraction=0.95))
    run_metadata = tf.RunMetadata()
    # session_config = None
    # run_metadata = None
    with tf.Session(config=session_config) as sess:
        tps = []
        for idx in range(outer_steps):
            sess.run(tf.global_variables_initializer())
            start_time = time.time()
            sess.run(benchmark_op, run_metadata=run_metadata)
            elapsed_time = time.time() - start_time
            tps.append(inner_steps * batch_size * seq_size / elapsed_time)
            logging.error("Iterations %d processed %f TPS.", idx, tps[-1])
        # Skip the first iteration where all the setup and allocation happens.
        tps = np.asarray(tps[1:])
        logging.error("Mean/Std/Max/Min throughput = %f / %f / %f / %f",
                      np.mean(tps), np.std(tps), tps.max(), tps.min())


def main(_):
    logging.set_verbosity(logging.INFO)
    benchmark()


if __name__ == "__main__":
    app.run(main)
