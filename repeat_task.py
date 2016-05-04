import tensorflow as tf
from tensorflow.models.rnn import rnn, rnn_cell
import numpy as np
import random
import ntm

# Define loss and optimizer
def var_seq_loss(preds, y, nsteps,seq_len,n_copies):
  start = seq_len + 2
  #start = tf.Print(start, [seq_len, n_copies,tf.shape(preds)])
  output_seq = tf.slice(
    preds,
    tf.pack([0, start, 0]),
    tf.pack([-1, seq_len*n_copies, -1]),
  )

  #output_seq = tf.Print(output_seq, [tf.shape(output_seq), tf.shape(y)])
  return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(output_seq, y))

def predict(out, nsteps, seq_len, n_copies):
  pred = tf.sigmoid(out)
  start = seq_len + 2
  rel_pred = tf.slice(
    pred,
    tf.pack([0, start, 0]),
    tf.pack([-1, seq_len*n_copies, -1]),
  )
  return rel_pred

def bits_err_per_seq(out, expected, nsteps, seq_len, n_copies):
  rel_pred = predict(out, nsteps, seq_len, n_copies)
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
  seq_len = tf.placeholder("int32")
  n_copies = tf.placeholder("int32")
  istate = tf.placeholder("float", [None, mem_nrow*mem_ncol + 2*mem_nrow])
  # TODO Remove after testing.
  #istate = tf.placeholder("float", [None, 600])
  nsteps = tf.placeholder("int32")
  ntm_cell = ntm.NTMCell(
      n_inputs=n_input,
      n_outputs=n_input,
      n_hidden=100,
      mem_nrows=mem_nrow,
      mem_ncols=mem_ncol,
      n_heads=1,
  )
  X = x
  # TODO Remove after testing.
  #lstm_cell = rnn_cell.BasicLSTMCell(
  #  num_units=100,
  #)
  #multi_cell = rnn_cell.MultiRNNCell([lstm_cell] * 3)
  #hidden = tf.Variable(tf.random_normal([n_input, 100], 0.1))
  #X = tf.reshape(X, [-1, n_input])
  #X = tf.nn.tanh(tf.matmul(X, hidden))
  #X = tf.reshape(X, [-1, max_steps, 100])
  outputs, states = rnn.dynamic_rnn(
      ntm_cell,
      X,
      #initial_state=istate,
      dtype=tf.float32,
      sequence_length=nsteps,
  )
  # TODO Remove after testing
  #hidden2 = tf.Variable(tf.random_normal([100, n_input], 0.1))
  #outputs = tf.reshape(outputs, [-1, 100])
  #outputs = tf.matmul(outputs, hidden2)
  #outputs = tf.reshape(outputs, [-1, max_steps, n_input])

  # Loss functions
  cost = var_seq_loss(outputs, y, nsteps,seq_len,n_copies)
  # Optimizer params as described in paper.
  opt = tf.train.RMSPropOptimizer(
    learning_rate=1e-4,
    momentum=0.9,
  )
  gvs = opt.compute_gradients(cost)
  # Gradient clipping as described in paper.
  clipped_gvs = []
  for g, v in gvs:
    # TODO Remove after testing
    #g = tf.Print(g, [g], v.name)
    clipped_gvs.append((tf.clip_by_value(g, -10, 10), v))
  optimizer = opt.apply_gradients(clipped_gvs)
  err = bits_err_per_seq(outputs, y, nsteps, seq_len, n_copies)

  return {
    'pred': predict(outputs, nsteps, seq_len, n_copies),
    'cost': cost,
    'optimizer': optimizer,
    'x': x,
    'y': y,
    'istate': istate,
    'steps': nsteps,
    'err': err,
    'states': states,
    'seq_len' : seq_len,
    'n_copies' : n_copies,
  }

def gen_seq(nseqs, max_steps, seq_len, nbits):
  n_copies = random.randint(2, 2**nbits - 1)
  n_copies_v = [int(digit) for digit in ('{0:0' + str(nbits) + 'b}').format(n_copies)]
  nsteps = seq_len +2 + seq_len*n_copies
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
    dummies = [zeros] * (seq_len*n_copies)
    x = seq + [n_copies_v]+[GO] + dummies + pad

    exp = seq * n_copies
    xs.append(x)
    ys.append(exp)
  return np.array(xs), np.array(ys), np.tile(nsteps, nseqs), seq_len, n_copies

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
    (xs, ys, nsteps, seq_len, n_copies) = gen_seq(
      nseqs=batch_size,
      max_steps=max_steps,
      seq_len=random.randint(seq_len_min, seq_len_max),
      nbits=n_input,
    )
    # TODO Remove after testing
    #istate = np.ones((batch_size, 600))
    sess.run(
      model['optimizer'],
      feed_dict={
        model['x']: xs,
        model['y']: ys,
        #model['istate']: istate,
        model['steps']: nsteps,
        model['seq_len'] : seq_len,
        model['n_copies'] : n_copies,
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
            model['seq_len'] : seq_len,
            model['n_copies'] : n_copies,
          },
        )
        print "Iter " + str(step*batch_size) + (
          ", seq_len= "+ str(seq_len)+ "n_copies= " + str(n_copies) + 
          ", Minibatch Loss= " + "{:.6f}".format(loss) +
          ", Average bit errors= " + "{:.6f}".format(err)
        )
    if step % 1000 == 0:
      save_path = saver.save(sess, "checkpoints/model" + str(step) + ".ckpt")
      print "Model saved in file: %s" % save_path

    step += 1
  print "Optimization Finished!"

# Training Parameters
max_steps = 128
seq_len_min = 1
seq_len_max = 5
batch_size = 10
n_input = 3
mem_nrow = 128
mem_ncol = 20
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

