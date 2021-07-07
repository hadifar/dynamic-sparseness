hz = 2048
block_size = 128
if (hz * 4 * hz * 2) % (block_size * block_size)!=0:
    raise Exception('fuck')
else:
    n_blocks = int((hz * 4 * hz * 2) / (block_size * block_size))

res = (hz * 2) % n_blocks == 0
print('number of blocks', int((hz * 4 * hz * 2) / (block_size * block_size)))
print('it is OK ?', res)
print(hz * 2)
print(int((hz * 4 * hz * 2) / (block_size * block_size)))

import tensorflow as tf

p = tf.random.truncated_normal([8192,4096])
m = tf.random.truncated_normal([1,128])



def block_mul(p, m):
    """p is large matrix"""
    p_x, p_y = p.shape  # (8192,4096)
    m_x, m_y = m.shape  # (1, 256)
    m_4d = tf.reshape(m, (m_x, 1, m_y, 1))
    m_broadcasted = tf.broadcast_to(m_4d, (m_x, p_x // m_x, m_y, p_y // m_y))
    mp = tf.reshape(m_broadcasted, (p_x, p_y))
    return mp


print(block_mul(p,m))