# To test features of tensorflow.

import tensorflow as tf
w = tf.constant(
    [[1, 0, 0],
    [0, 1, 0]])
M = tf.constant(
  [
    [
      [1, 2],
      [3, 4],
      [5, 6]
    ],
    [
      [7, 8],
      [9, 10],
      [11, 12]
    ],
  ]
)
w = tf.expand_dims(w, 1)
test1 = tf.batch_matmul(w, M)

with tf.Session() as sess:
  print sess.run(test1)
