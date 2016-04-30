import tensorflow as tf
from tensorflow.models.rnn import rnn
import ntm

X = [
  tf.constant([
    [1., 2.],
    [7., 8.],
  ]),
  tf.constant([
    [3., 4.],
    [9., 10.],
  ]),
  tf.constant([
    [5., 6.],
    [11., 12.],
  ]),
]

ntm_cell = ntm.NTMCell(
  n_inputs=2,
  n_outputs=3,
  n_hidden=4,
  mem_nrows=10,
  mem_ncols=5,
  n_heads=1,
)

outputs, _ = rnn.rnn(ntm_cell, X, dtype=tf.float32)

# Simple test to make sure it runs.
with tf.Session() as sess:
  sess.run(tf.initialize_all_variables())
  for out in outputs:
    print sess.run(out)
