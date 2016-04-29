import tensorflow as tf
from tensorflow.models.rnn import rnn
import numpy as np
import random
import ntm

# Define loss and optimizer
def var_seq_loss(preds, y, nsteps):
  seq_len = (nsteps[0] - 1) / 2
  start = seq_len + 1
  output_seq = tf.slice(
    preds,
    tf.pack([0, start, 0]),
    tf.pack([-1, seq_len, -1]),
  )
  return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(output_seq, y))

def predict(preds):
  return tf.sigmoid(preds)

def create_rnn(max_steps, n_input):
  # Batch size, max_steps, n_input
  x = tf.placeholder("float", [None, None, n_input])
  y = tf.placeholder("float", [None, None, n_input])
  istate = tf.placeholder("float", [None, 128*20+128])
  nsteps = tf.placeholder("int32")
  # TODO Expose params.
  ntm_cell = ntm.NTMCell(
      n_inputs=n_input,
      n_hidden=100,
      mem_nrows=128,
      mem_ncols=20,
  )
  outputs, _ = rnn.dynamic_rnn(
      ntm_cell,
      x,
      initial_state=istate,
      sequence_length=nsteps,
  )

  # Loss functions
  cost = var_seq_loss(outputs, y, nsteps)
  # Optimizer params as described in paper.
  opt = tf.train.RMSPropOptimizer(
    learning_rate=1e-4,
    momentum=0.9,
  )
  gvs = opt.compute_gradients(cost)
  # Gradient clipping as described in paper.
  clipped_gvs = [
    (tf.clip_by_value(g, -10, 10), v)
    for g, v in gvs
  ]
  optimizer = opt.apply_gradients(clipped_gvs)

  return {
    'pred': outputs,
    'cost': cost,
    'optimizer': optimizer,
    'x': x,
    'y': y,
    'istate': istate,
    'steps': nsteps,
  }

def gen_seq(nseqs, max_steps, seq_len, nbits):
  nsteps = 2*seq_len + 1
  assert nsteps <= max_steps
  zeros = [0] * nbits
  ones = [1] * nbits
  xs = []
  ys = []
  for _ in xrange(nseqs):
    # Note that we reserve the all-zero and all-one vectors as the 
    # padding and delimeter symbols
    seq = [random.randint(1, 2**nbits - 1) for _ in xrange(seq_len)]
    # Convert each int to binary
    seq = [
            [int(digit) for digit in '{0:08b}'.format(num)]
            for num in seq
          ]
    npad = max_steps - nsteps
    pad = [zeros] * npad
    # Dummy inputs after the delimeter / outputs before the delimiter
    dummies = [zeros] * seq_len
    x = seq + [ones] + dummies + pad
    xs.append(x)
    ys.append(seq)
  return np.array(xs), np.array(ys), np.tile(nsteps, nseqs)

def train(
    model,
    n_input,
    max_steps,
    training_iters=1000000,
    batch_size=128,
    display_step=10,
    ):
  sess = tf.Session()
  # TODO Too slow!
  #merged = tf.merge_all_summaries()
  #train_writer = tf.train.SummaryWriter("log", sess.graph)

  # Initializing the variables
  init = tf.initialize_all_variables()
  sess.run(init)
  step = 1

  # We generate a finite amount of training data even
  # though we could theoretically have infinite data
  # for debugging purposes (loss should almost always decrease).
  training_data = []
  nbatches = 10
  for i in xrange(nbatches):
    seq_len = random.randint(18, 20)
    (xs, ys, nsteps) = gen_seq(
      nseqs=batch_size,
      max_steps=max_steps,
      seq_len=seq_len,
      nbits=n_input,
    )
    training_data.append((xs, ys, nsteps))
  batch_idx = 0

  print "Training commencing..."
  while step * batch_size < training_iters:
    xs, ys, nsteps = training_data[batch_idx]
    batch_idx = (batch_idx + 1) % nbatches
    # TODO Is this the right thing to do?
    # TODO Refactor magic number
    istate = np.random.uniform(size=(batch_size, 128*20+128))
    sess.run(
      model['optimizer'],
      feed_dict={
        model['x']: xs,
        model['y']: ys,
        model['istate']: istate,
        model['steps']: nsteps,
      },
    )
    if step % display_step == 0:
        loss = sess.run(
          model['cost'],
          feed_dict={
            model['x']: xs,
            model['y']: ys,
            model['istate']: istate,
            model['steps']: nsteps,
          },
        )
        print "Iter " + str(step*batch_size) + (
          ", Minibatch Loss= " + "{:.6f}".format(loss))
        # TODO Too slow!
        #train_writer.add_summary(summary, step)

    step += 1
  print "Optimization Finished!"

# Training Parameters
max_steps = 41
n_input = 8
model = create_rnn(max_steps, n_input)

if __name__ == "__main__":
  train(
    model=model,
    n_input=n_input,
    max_steps=max_steps,
  )
