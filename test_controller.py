import tensorflow as tf
import ntm

ntm_cell = ntm.NTMCell(
  n_inputs=1,
  n_hidden=2,
  mem_nrows=10,
  mem_ncols=2,
)

output, head = ntm_cell.controller(
  inputs=[[1.], [2.], [3.]],
  read=[[1., 2.], [3., 4.], [5., 6.]],
)

# Simple test to make sure it compiles and runs.
with tf.Session() as sess:
  sess.run(tf.initialize_all_variables())
  print sess.run(output)
  for name, tens in head.iteritems():
    print name
    print sess.run(tens)
