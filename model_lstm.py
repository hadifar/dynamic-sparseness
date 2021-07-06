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

import tensorflow.compat.v1 as tf

# import sys
# sys.path.append('..')
# from sgk.sparse.ops.backend import kernels

tf.compat.v1.disable_eager_execution()
tf.disable_v2_behavior()


class DynamicSparseGate(tf.Module):
    def __init__(self, hz, sparsity, block_size=4):
        assert hz % block_size == 0, print("hidden size is {}, block-size is {}".format(hz,block_size))
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

    def __call__(self, inps, dense_weight):
        inps = tf.expand_dims(tf.reduce_mean(inps, axis=tf.range(tf.rank(inps) - 1, )), 0)
        gval = tf.matmul(inps, self.g_weight)
        topval = tf.math.top_k(gval, self.top_blocks)[0][:, -1]  # smallest value in topk
        expandval = tf.cast(gval >= topval, tf.float32) * gval

        matrix = self.gate_loc(expandval)
        # values = tf.boolean_mask(matrix, mask)
        denominator = tf.math.reduce_sum(matrix, axis=-1, keepdims=True) / tf.cast(matrix.shape[-1], dtype=tf.float32)
        matrix = matrix / denominator

        # return matrix
        return self.d2s(matrix, dense_weight)

    @tf.function
    def gate_loc(self, m):
        """p is large matrix"""
        p_x, p_y = self.hz * 4, self.hz * 2
        m_x, m_y = m.shape
        m_4d = tf.reshape(m, (m_x, 1, m_y, 1))
        m_broadcasted = tf.broadcast_to(m_4d, (m_x, p_x // m_x, m_y, p_y // m_y))
        mp = tf.reshape(m_broadcasted, (p_x, p_y))
        return mp

    @tf.function
    def d2s(self, matrix, dense_weight):
        ###############################################################
        ###############################################################
        # Extract the nonzero values.
        # values = matrix.compress((matrix != 0).flatten())
        mask = tf.math.not_equal(matrix, 0)
        values = tf.boolean_mask(dense_weight, mask)
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
        return self._rows, self._columns, values, row_indices, row_offsets, column_indices


class BaseModel(tf.Module):
    def __init__(self, hz, vocab_size=30000):
        self.hidden_size = hz
        self.vocab_size = vocab_size

        # embedding
        self.embed_weight = tf.get_variable('emb', [vocab_size, hz])

        # LSTM
        self.rnn_weight = tf.get_variable(initializer=tf.truncated_normal([hz * 4, hz + hz], stddev=0.1),
                                          name='x_weight',
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


class LSTMModel(BaseModel):
    def __init__(self, hz, vocab_size=30000, block_size=128, sparsity=0.95):
        super().__init__(hz, vocab_size)

        # dynamic gate
        self.dynamic_gate = DynamicSparseGate(hz, sparsity=sparsity, block_size=block_size)

    # @tf.function
    def lstm(self, inps):
        rnn_weight = self.rnn_weight * self.dynamic_gate(inps, None)
        # rows, columns, values, row_indices, row_offsets, column_indices = \
        #     self.dynamic_gate(inps, self.rnn_weight)

        init_state = tf.zeros(shape=[2, inps.shape[0], self.hidden_size], dtype=tf.float32)

        def step(hprev, x):
            st_1, ct_1 = tf.unstack(hprev)
            # st_1, x = tf.transpose(st_1), tf.transpose(x)

            fc_gate = tf.matmul(rnn_weight, tf.transpose(tf.concat([x, st_1], axis=1)))
            # fc_gate = kernels.spmm(rows, columns, values, row_indices, row_offsets, column_indices, x, False, False)

            fc_gate = tf.transpose(fc_gate) + self.bias
            i, f, g, o = tf.split(fc_gate, 4, axis=1)
            i, f, g, o = tf.sigmoid(i), tf.sigmoid(f), tf.tanh(g), tf.sigmoid(o)
            ct = ct_1 * f + g * i
            st = tf.tanh(ct) * o

            return tf.stack([st, ct])

        states = tf.scan(step, tf.transpose(inps, [1, 0, 2]), initializer=init_state)

        return tf.transpose(states, [1, 2, 0, 3])[0]


#
# class LSTMSparseModel(BaseModel):
#     def __init__(self, hz, vocab_size=30000, block_size=4, sparsity=0.95):
#         super().__init__(hz, vocab_size)
#
#         # dynamic gate
#         self.dynamic_gate = DynamicSparseGate(hz, sparsity=sparsity, block_size=block_size)
#
#     @tf.function
#     def lstm(self, inps):
#         new_x_weight, new_h_weight, bias_weight = self.x_weight, self.h_weight, self.bias
#         mask, rows, columns, row_indices, row_offsets, column_indices = self.dynamic_gate(inps)
#
#         values2 = tf.boolean_mask(new_h_weight, mask)
#         init_state = tf.zeros(shape=[2, inps.shape[0], self.hidden_size], dtype=tf.float32)
#
#         def step(hprev, x):
#             st_1, ct_1 = tf.unstack(hprev)
#             st_1, x = tf.transpose(st_1), tf.transpose(x)
#
#             # fc_gate = tf.matmul(new_h_weight, st_1) + tf.matmul(new_x_weight, x)
#             fc_gate = \
#                 kernels.spmm(rows, columns, values1, row_indices, row_offsets, column_indices, x, False, False) + \
#                 kernels.spmm(rows, columns, values2, row_indices, row_offsets, column_indices, st_1, False, False)
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
#
#
# model = LSTMSparseModel(hz=1024, block_size=128, sparsity=0.5)
# model = LSTMModel(hz=1024)
#
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     dummy_input = tf.random.uniform([10, 512], minval=0, maxval=30000, dtype=tf.int32)
#     print(sess.run(model.loss(model(dummy_input), dummy_input)))
