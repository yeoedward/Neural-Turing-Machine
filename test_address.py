import tensorflow as tf
import ntm

wg_i = tf.constant([1, 2, 3, 4, 5, 6, 7])
shift_i = tf.constant([0, 1, 1])
row_idx = tf.constant(1)
test1 = ntm.NTMCell.conv_sum(wg_i, shift_i, row_idx)
with tf.Session() as sess:
  assert sess.run(test1) == 21

wg = tf.constant([
  [1, 2, 3, 4, 5, 6, 7],
  [8, 9, 10, 11, 12, 13, 14],
])
shift = tf.constant([
  [0, 1, 1], 
  [-1, 0, 1],
])
test2 = ntm.NTMCell.rotate(wg, shift)

with tf.Session() as sess:
  res = sess.run(test2)
  # Check a few of the values by hand
  assert res[0][1] == 21
  assert res[1][1] == 2
  assert res[1][3] == -12
