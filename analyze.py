# Print bitmaps of inputs and outputs for sequences of various length.
# For the report.
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from copy_task import model, gen_seq
from ntm import NTMCell

test_lens = [5, 10, 30, 40, 50]
inputs = []
outputs = []
nsteps = []
for seq_len in test_lens:
  max_steps = seq_len*2 + 1
  x, y, steps = gen_seq(1, max_steps, seq_len, 3)
  inputs.append(x)
  outputs.append(y)
  nsteps.append(steps)

saver = tf.train.Saver()
with tf.Session() as sess:
  # Load saved model
  saver.restore(sess, "checkpoints/copy_task/model41000.ckpt")
  for i in [2]:#xrange(len(inputs)):
    output = sess.run(model["pred"], feed_dict={
      model['x']: inputs[i],
      model['steps']: nsteps[i],
    })
    #seq_len = test_lens[i]
    #plt.matshow(np.transpose(inputs[i][0][:seq_len]))
    #plt.savefig('images/inputs%d.png' % seq_len, dpi=100)
    #plt.matshow(np.transpose(output[0]))
    #plt.savefig('images/outputs%d.png' % seq_len, dpi=100)
