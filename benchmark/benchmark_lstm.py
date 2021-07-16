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

"adopted from: https://github.com/google-research/google-research/tree/master/sgk"

import os
import warnings

from scipy.sparse import random

warnings.filterwarnings("ignore", message=r"Passing", category=FutureWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import abc
import math
import time

import numpy as np
import tensorflow.compat.v1 as tf
from absl import app
from absl import logging
from absl import flags
import itertools

from sgk.sparse.ops.backend import kernels

tf.compat.v1.disable_eager_execution()
tf.disable_v2_behavior()

FLAGS = flags.FLAGS

flags.DEFINE_string("config", "sparse", "Whether to run sparse or dense.")

flags.DEFINE_integer("inner_steps", 10, "Benchmark steps for inner loop.")

flags.DEFINE_integer("outer_steps", 10, "Benchmark steps for outer loop.")


def _dense_to_sparse(matrix):
    """Converts dense numpy matrix to a csr sparse matrix."""
    assert len(matrix.shape) == 2

    # Extract the nonzero values.
    values = matrix.compress((matrix != 0).flatten())

    # Calculate the offset of each row.
    mask = (matrix != 0).astype(np.int32)
    row_offsets = np.concatenate(([0], np.cumsum(np.add.reduce(mask, axis=1))),
                                 axis=0)

    # Create the row indices and sort them.
    row_indices = np.argsort(-1 * np.diff(row_offsets))

    # Extract the column indices for the nonzero values.
    x = mask * (np.arange(matrix.shape[1]) + 1)
    column_indices = x.compress((x != 0).flatten())
    column_indices = column_indices - 1

    # Cast the desired precision.
    values = values.astype(np.float32)
    row_indices, row_offsets, column_indices = [
        x.astype(np.uint32) for x in
        [row_indices, row_offsets, column_indices]
    ]
    return values, row_indices, row_offsets, column_indices


class DynamicSparseGate(tf.Module):
    def __init__(self, hz, sparsity=0.5, block_size=4):
        assert hz % block_size == 0
        assert (hz * 4 * hz * 2) % (block_size * block_size) == 0

        self.sparsity = sparsity
        self.hz = hz
        self.block_size = block_size
        self.n_blocks = int((hz * 4 * hz * 2) / (block_size * block_size))
        self.top_blocks = int(math.ceil((1. - sparsity) * self.n_blocks))

        print('number of hidden units ----> {} \n'
              'number of blocks ----------> {} \n'
              'number of active blocks ---> {} \n'
              'sparsity level ------------> {} \n'
              ''.format(self.hz, self.n_blocks, self.top_blocks, sparsity)
              )

        self.g_weight = tf.get_variable(initializer=tf.truncated_normal([hz, self.n_blocks], stddev=0.1),
                                        name='gate_weight',
                                        dtype=tf.float32)

        rand_bsr = np.array(random(4 * hz, 2 * hz, density=1. - sparsity).A != 0).astype(np.int)
        values, row_indices_, row_offsets_, column_indices_ = _dense_to_sparse(rand_bsr)

        self._rows = tf.Variable(
            hz * 4,
            trainable=False,
            name="dsg_rows",
            dtype=tf.int32)
        self._columns = tf.Variable(
            hz * 2,
            trainable=False,
            name="dsg_columns",
            dtype=tf.int32)

        self._row_indices = tf.get_variable(
            initializer=row_indices_,
            trainable=False,
            name="row_indices",
            dtype=tf.uint32)
        self._row_offsets = tf.get_variable(
            initializer=row_offsets_,
            trainable=False,
            name="row_offsets",
            dtype=tf.uint32)
        self._column_indices = tf.get_variable(
            initializer=column_indices_,
            trainable=False,
            name="column_indices",
            dtype=tf.uint32)

        self.values = tf.get_variable(
            initializer=values,
            trainable=False,
            name="values",
            dtype=tf.float32)

    def __call__(self, inps):
        gval = tf.matmul(inps, self.g_weight)
        topval = tf.math.top_k(gval, self.top_blocks)[0][:, -1]
        gate = tf.cast((gval >= topval), tf.float32) * gval
        gate = tf.repeat(gate, self.block_size ** 2, axis=-1)
        gate = tf.reshape(gate, (self.hz * 4, self.hz * 2))
        gate = gate / (tf.math.reduce_sum(gate) / tf.cast(tf.size(gate), tf.float32))
        # we exclude the cost of dense to sparse
        # we add it to Graph with dummy operation tf.reduce_mean(gate)
        return self._rows, self._columns, self.values + tf.reduce_mean(
            gate), self._row_indices, self._row_offsets, self._column_indices

    @tf.function
    def d2s(self, matrix, weight):
        """dense to sparse operation"""

        # Extract the nonzero values.
        mask = tf.math.not_equal(matrix, 0)
        values = tf.boolean_mask(weight, mask)

        # Calculate offset of each row
        mask = tf.cast(mask, tf.int32)
        row_offsets = tf.concat([[0], tf.cumsum(tf.reduce_sum(mask, axis=1))], axis=0)

        # Create the row indices and sort them.
        row_indices = tf.argsort(-(row_offsets[1:] - row_offsets[:-1]))

        # Extract the column indices for the nonzero values.
        x = mask * (tf.range(matrix.shape[1]) + 1)
        column_indices = tf.boolean_mask(x, tf.math.not_equal(matrix, 0))
        column_indices = column_indices - 1

        row_indices = tf.cast(row_indices, tf.uint32)
        row_offsets = tf.cast(row_offsets, tf.uint32)
        column_indices = tf.cast(column_indices, tf.uint32)
        return self._rows, self._columns, values, row_indices, row_offsets, column_indices


class BaseModel(tf.Module):
    def __init__(self, hz, vocab_size=30000):
        self.hidden_size = hz
        self.vocab_size = vocab_size

        # embedding
        self.embed_weight = tf.get_variable('emb', [vocab_size, hz])

        # LSTM
        self.recurrent_weight = tf.get_variable(initializer=tf.truncated_normal([hz * 4, hz * 2], stddev=0.1),
                                                name='recurrent_weight',
                                                dtype=tf.float32)

        self.recurrent_bias = tf.get_variable(initializer=tf.truncated_normal([hz * 4], stddev=0.1),
                                              name='recurrent_bias',
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
        weight, bias = self.recurrent_weight, self.recurrent_bias
        init_state = tf.zeros(shape=[2, inps.shape[0], self.hidden_size], dtype=tf.float32)

        def step(hprev, x):
            st_1, ct_1 = tf.unstack(hprev)

            fc_gate = tf.matmul(weight, tf.transpose(tf.concat([x, st_1], -1)))
            fc_gate = tf.transpose(fc_gate) + bias
            i, f, g, o = tf.split(fc_gate, 4, axis=1)
            i, f, g, o = tf.sigmoid(i), tf.sigmoid(f), tf.tanh(g), tf.sigmoid(o)
            ct = ct_1 * f + g * i
            st = tf.tanh(ct) * o

            return tf.stack([st, ct])

        states = tf.scan(step, tf.transpose(inps, [1, 0, 2]), initializer=init_state)

        return tf.transpose(states, [1, 2, 0, 3])[0]


class ToySparseModel(BaseModel):
    def __init__(self, hz, block_size=4, sparsity=0.95, vocab_size=30000):
        super().__init__(hz, vocab_size)

        # dynamic gate
        self.dynamic_gate = DynamicSparseGate(hz, sparsity=sparsity, block_size=block_size)

    @tf.function
    def lstm(self, inps):
        weight, bias = self.recurrent_weight, self.recurrent_bias

        init_state = tf.zeros(shape=[2, inps.shape[0], self.hidden_size], dtype=tf.float32)

        def step(hprev, x):
            st_1, ct_1 = tf.unstack(hprev)
            rows, columns, values, row_indices, row_offsets, column_indices = self.dynamic_gate(x)

            fc_gate = kernels.spmm(rows,
                                   columns,
                                   values,
                                   row_indices,
                                   row_offsets,
                                   column_indices,
                                   tf.transpose(tf.concat([x, st_1], -1)), False, False)

            fc_gate = tf.transpose(fc_gate) + bias

            i, f, g, o = tf.split(fc_gate, 4, axis=1)
            i, f, g, o = tf.sigmoid(i), tf.sigmoid(f), tf.tanh(g), tf.sigmoid(o)
            ct = ct_1 * f + g * i
            st = tf.tanh(ct) * o

            return tf.stack([st, ct])

        states = tf.scan(step, tf.transpose(inps, [1, 0, 2]), initializer=init_state)

        return tf.transpose(states, [1, 2, 0, 3])[0]


def benchmark(model, hidden_size, seq_size, sparsity, block_size):
    """Run repeatedly on dummy data to benchmark inference."""
    # Turn off Grappler optimizations.
    options = {"disable_meta_optimizer": True}
    tf.config.optimizer.set_experimental_options(options)

    # Run only the model body (no data pipeline) on device.
    dummy_input = tf.random.uniform([1, seq_size], minval=0, maxval=30000, dtype=tf.int32)
    # feature_shape = [batch_size, image_size, hidden_size]
    # features = {"targets": tf.zeros(feature_shape, dtype=tf.float32)}

    # warm-up
    model.loss(model(dummy_input), dummy_input)

    def call_model(features):
        logits = model(features)
        return model.loss(logits, features)

    # Run the function body in a loop to amortize session overhead.
    loop_index = tf.zeros([], dtype=tf.int32)
    initial_loss = tf.zeros([])

    def loop_cond(idx, _):
        return tf.less(idx, tf.constant(FLAGS.inner_steps, dtype=tf.int32))

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
        for idx in range(FLAGS.outer_steps):
            sess.run(tf.global_variables_initializer())
            start_time = time.time()
            sess.run(benchmark_op, run_metadata=run_metadata)
            elapsed_time = time.time() - start_time
            tps.append(FLAGS.inner_steps * seq_size / elapsed_time)
            # logging.error("Iterations %d processed %f TPS.", idx, tps[-1])
        # Skip the first iteration where all the setup and allocation happens.
        tps = np.asarray(tps[1:])
        logging.error("hz: %f, seq %f, sparsity %f, block_size %f,  Mean/Std/Max/Min throughput = %f / %f / %f / %f",
                      hidden_size, seq_size, sparsity, block_size, np.mean(tps), np.std(tps), tps.max(), tps.min())


def main(_):
    # model params
    logging.set_verbosity(logging.ERROR)

    hidden_sizes = [1024 * 6, 1024 * 4, 1024 * 2]
    seq_sizes = [128]
    sparsity = [0.1, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
    block_sizes = [128]

    if FLAGS.config == 'sparse':
        for hz, seq, sp, blz in list(itertools.product(*[hidden_sizes, seq_sizes, sparsity, block_sizes])):
            tf.reset_default_graph()
            model = ToySparseModel(hz=hz, block_size=blz, sparsity=sp)
            benchmark(model, hz, seq, sp, blz)
    else:
        for hz, seq, sp, blz in list(itertools.product(*[hidden_sizes, seq_sizes, [1.], [1]])):
            tf.reset_default_graph()
            model = ToyModel(hz=hz)
            benchmark(model, hz, seq, sp, blz)


if __name__ == "__main__":
    app.run(main)
