import tensorflow as tf
ntm_rotate_module = tf.load_op_library(
  "tensorflow/bazel-bin/tensorflow/core/user_ops/rotate.so",
)
weights = tf.constant([
  [1., 2., 3., 4., 5., 6., 7.],
  [8., 9., 10., 11., 12., 13., 14.],
])
shifts = tf.constant([
  [0., 1., 1.], 
  [-1., 0., 1.],
])

with tf.Session() as sess:
  res = sess.run(ntm_rotate_module.ntm_rotate(weights, shifts))
  # Check a few of the values by hand
  assert res[0][1] == 21.
  assert res[1][1] == 2.
  assert res[1][3] == -12.
  print res
