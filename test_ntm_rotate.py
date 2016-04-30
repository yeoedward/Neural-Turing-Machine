import math
import tensorflow as tf
import tensorflow.python.user_ops as ntm_rotate_module
import ntm
ntm_rotate_module = tf.load_op_library(
  "tensorflow/bazel-bin/tensorflow/core/user_ops/rotate.so",
)

# Citation: https://github.com/carpedm20/NTM-tensorflow/blob/master/ops.py
# Used for testing purposes. Too slow if batch_size is not small.
def conv(v, k):
  """Computes circular convolution.
  Args:
      v: a 1-D `Tensor` (vector)
      k: a 1-D `Tensor` (kernel)
  """
  size = int(v.get_shape()[0])
  kernel_size = int(k.get_shape()[0])
  kernel_shift = int(math.floor(kernel_size/2.0))

  def loop(idx):
      if idx < 0: return size + idx
      if idx >= size : return idx - size
      else: return idx

  kernels = []
  for i in xrange(size):
      indices = [loop(i+j) for j in xrange(kernel_shift, -kernel_shift-1, -1)]
      v_ = tf.gather(v, indices)
      kernels.append(tf.reduce_sum(v_ * k, 0))

  # # code with double loop
  # for i in xrange(size):
  #     for j in xrange(kernel_size):
  #         idx = i + kernel_shift - j + 1
  #         if idx < 0: idx = idx + size
  #         if idx >= size: idx = idx - size
  #         w = tf.gather(v, int(idx)) * tf.gather(kernel, j)
  #         output = tf.scatter_add(output, [i], tf.reshape(w, [1, -1]))

  return tf.pack(kernels)

def rotate(v, k):
  nbatches = int(v.get_shape()[0])
  res = []
  for i in xrange(nbatches):
    res.append(conv(
      tf.squeeze(tf.slice(v, [i, 0], [1, -1])),
      tf.squeeze(tf.slice(k, [i, 0], [1, -1])),
    ))
  return tf.pack(res)

weights = tf.constant([
  [1., 2., 3., 4., 5., 6., 7.],
  [8., 9., 10., 11., 12., 13., 14.],
])
shifts = tf.constant([
  [0., 0., 1.], 
  [1., 0., 0.],
])

with tf.Session() as sess:
  print sess.run(rotate(weights, shifts))
  print sess.run(ntm_rotate_module.ntm_rotate(weights, shifts))
