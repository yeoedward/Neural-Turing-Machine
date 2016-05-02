import tensorflow as tf
from tensorflow.models.rnn import rnn, rnn_cell
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

def predict(out, nsteps):
  pred = tf.sigmoid(out)
  seq_len = (nsteps[0] - 1) / 2
  start = seq_len + 1
  rel_pred = tf.slice(
    pred,
    tf.pack([0, start, 0]),
    tf.pack([-1, seq_len, -1]),
  )
  return rel_pred

def bits_err_per_seq(out, expected, nsteps):
  rel_pred = predict(out, nsteps)
  rel_pred = tf.Print(
    rel_pred,
    [tf.slice(rel_pred, [0, 0, 0], [1, -1, 1])],
    "predicted",
    summarize=20,
  )
  expected = tf.Print(
    expected,
    [tf.slice(expected, [0, 0, 0], [1, -1, 1])],
    "expected",
    summarize=20,
  )
  diff = rel_pred - expected
  return tf.reduce_mean(tf.reduce_sum(tf.abs(diff), [1, 2]))

def create_rnn(max_steps, n_input, mem_nrow, mem_ncol):
  # Batch size, max_steps, n_input
  x = tf.placeholder("float", [None, None, n_input])
  y = tf.placeholder("float", [None, None, n_input])
  nsteps = tf.placeholder("int32")
  ntm_cell = ntm.NTMCell(
      n_inputs=n_input,
      n_outputs=n_input,
      n_hidden=100,
      mem_nrows=mem_nrow,
      mem_ncols=mem_ncol,
      n_heads=1,
  )
  outputs, _ = rnn.dynamic_rnn(
      ntm_cell,
      x,
      dtype=tf.float32,
      sequence_length=nsteps,
  )

  # Loss measures
  cost = var_seq_loss(outputs, y, nsteps)
  err = bits_err_per_seq(outputs, y, nsteps)

  # Optimizer params as described in paper.
  opt = tf.train.RMSPropOptimizer(
    learning_rate=1e-4,
    momentum=0.9,
  )
  # Gradient clipping as described in paper.
  gvs = opt.compute_gradients(cost)
  clipped_gvs = []
  for g, v in gvs:
    clipped_gvs.append((tf.clip_by_value(g, -10, 10), v))
  optimizer = opt.apply_gradients(clipped_gvs)

  return {
    'x': x,
    'y': y,
    'steps': nsteps,
    'cost': cost,
    'err': err,
    'optimizer': optimizer,
    'pred': predict(outputs, nsteps),
  }

def gen_seq(nseqs, max_steps, seq_len, nbits):
  nsteps = 2*seq_len + 1
  assert nsteps <= max_steps
  zeros = [0] * nbits
  GO = [0] * (nbits - 1) + [1]
  xs = []
  ys = []
  for _ in xrange(nseqs):
    # Note that we reserve the 0 and 1 in binary as the 
    # padding and delimeter symbols
    seq = [random.randint(2, 2**nbits - 1) for _ in xrange(seq_len)]
    # Convert each int to binary
    seq = [
            [int(digit) for digit in ('{0:0' + str(nbits) + 'b}').format(num)]
            for num in seq
          ]
    npad = max_steps - nsteps
    pad = [zeros] * npad
    # Dummy inputs after the delimeter / outputs before the delimiter
    dummies = [zeros] * seq_len
    x = seq + [GO] + dummies + pad
    xs.append(x)
    ys.append(seq)
  return np.array(xs), np.array(ys), np.tile(nsteps, nseqs)

def train(
    model,
    n_input,
    max_steps,
    training_iters=1e8,
    batch_size=128,
    display_step=10,
    seq_len_min=1,
    seq_len_max=20,
    ):
  sess = tf.Session()
  # Initializing the variables
  init = tf.initialize_all_variables()
  sess.run(init)
  step = 1

  saver = tf.train.Saver()
  print "Training commencing..."
  while step * batch_size < training_iters:
    (xs, ys, nsteps) = gen_seq(
      nseqs=batch_size,
      max_steps=max_steps,
      seq_len=random.randint(seq_len_min, seq_len_max),
      nbits=n_input,
    )
    sess.run(
      model['optimizer'],
      feed_dict={
        model['x']: xs,
        model['y']: ys,
        #model['istate']: istate,
        model['steps']: nsteps,
      },
    )
    if step % display_step == 0:
        err, loss = sess.run(
          [model['err'], model['cost']],
          feed_dict={
            model['x']: xs,
            model['y']: ys,
            #model['istate']: istate,
            model['steps']: nsteps,
          },
        )
        print "Iter " + str(step*batch_size) + (
          ", Minibatch Loss= " + "{:.6f}".format(loss) +
          ", Average bit errors= " + "{:.6f}".format(err)
        )
    if step % 1000 == 0:
      save_path = saver.save(sess, "checkpoints/model" + str(step) + ".ckpt")
      print "Model saved in file: %s" % save_path

    step += 1

  print "Optimization Finished!"

# Training Parameters
max_steps = 11
seq_len_min = 1
seq_len_max = 5
batch_size = 1
n_input = 3
mem_nrow = 10
mem_ncol = 5
model = create_rnn(
  max_steps=max_steps,
  n_input=n_input,
  mem_nrow=mem_nrow,
  mem_ncol=mem_ncol,
)

if __name__ == "__main__":
  train(
    model=model,
    n_input=n_input,
    max_steps=max_steps,
    seq_len_min=seq_len_min,
    seq_len_max=seq_len_max,
    batch_size=batch_size,
  )
