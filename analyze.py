# Analyze specfic test case

import tensorflow as tf
from copy_task import model
from ntm import NTMCell

saver = tf.train.Saver()
with tf.Session() as sess:
  # Load saved model
  saver.restore(sess, "checkpoints/model4000.ckpt")
  # 2675435 in binary.
  xs = [[
    [0, 1, 0],
    [1, 1, 0],
    [1, 1, 1],
    [1, 0, 1],
    [1, 0, 0],
    [0, 1, 1],
    [1, 0, 1],
    [0, 0, 1], # GO
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
  ]]
  nsteps = [15]
  print sess.run(model["pred"], feed_dict={
    model['x']: xs,
    model['steps']: nsteps,
  })
