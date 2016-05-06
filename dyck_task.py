import tensorflow as tf
from tensorflow.models.rnn import rnn, rnn_cell
import numpy as np
import random
import ntm

def oparen(nbits):
  return to_binary(2**nbits - 1, nbits)
def cparen(nbits):
  return to_binary(2**nbits - 2, nbits)

def to_binary(num, nbits):
  return [int(digit) for digit in ('{0:0' + str(nbits) + 'b}').format(num)]

def dyck_word(seq_len, nbits):
  assert seq_len % 2 == 0
  # S -> epsilon
  if seq_len == 0: return []
  # S -> (S)
  if random.randint(0, 1) == 0:
    return [oparen(nbits)] + dyck_word(seq_len - 2, nbits) + [cparen(nbits)]
  # S -> SS
  first_len = 2 * random.randint(0, seq_len / 2)
  second_len = seq_len - first_len
  return dyck_word(first_len, nbits) + dyck_word(second_len, nbits)

def is_dyck_word(seq, nbits):
  counter = 0
  for c in seq:
    if c == oparen(nbits):
      counter += 1
    elif c == cparen(nbits):
      counter -= 1
      if counter < 0: return 0
    else: # Non-paren character.
      return 0
  return 1 if counter == 0 else 0

def gen_seq(nseqs, max_steps, seq_len, nbits):
  nsteps = seq_len + 2
  assert nsteps <= max_steps
  assert seq_len % 2 == 0
  zeros = [0] * nbits
  GO = [0] * (nbits - 1) + [1]
  xs = []
  ys = []
  for _ in xrange(nseqs):
    if random.randint(0, 1) == 0:
      # Generate dyck word.
      # We don't want to leave it to randomness as our training data
      # might have too many negative examples.
      seq = dyck_word(seq_len, nbits)
      y = 1
    else:
      # Generate random word, might or might not be dyck word.
      # Note that we reserve the 0 and 1 in binary as the 
      # padding and delimeter symbols.
      seq = [random.randint(2, 2**nbits - 1) for _ in xrange(seq_len)]
      # Convert each int to binary
      seq = [
              to_binary(num, nbits)
              for num in seq
            ]
      y = is_dyck_word(seq, nbits)
    npad = max_steps - nsteps
    pad = [zeros] * npad
    x = seq + [GO] + [zeros] + pad
    xs.append(x)
    ys.append(y)
  return np.array(xs), np.array(ys), np.tile(nsteps, nseqs)

# Define loss and optimizer
def var_seq_loss(preds, y, nsteps):
  start = nsteps[0] - 1
  output_seq = tf.slice(
    preds,
    tf.pack([0, start, 0]),
    tf.pack([-1, 1, -1]),
  )
  return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(output_seq, y))

def predict(out, nsteps, y):
  pred = tf.sigmoid(out)
  start = nsteps[0] - 1
  rel_pred = tf.slice(
    pred,
    tf.pack([0, start, 0]),
    tf.pack([-1, 1, -1]),
  )
  rel_pred = tf.Print(rel_pred, [y], "Expected")
  rel_pred = tf.Print(rel_pred, [rel_pred], "Predicted")
  return rel_pred

def create_rnn(max_steps, n_input, mem_nrow, mem_ncol):
  # Batch size, max_steps, n_input
  x = tf.placeholder("float", [None, None, n_input])
  y = tf.placeholder("float")
  nsteps = tf.placeholder("int32")
  ntm_cell = ntm.NTMCell(
      n_inputs=n_input,
      n_outputs=1,
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
    'optimizer': optimizer,
    'pred': predict(outputs, nsteps, y),
  }

def train(
    model,
    n_input,
    max_steps,
    batch_size,
    seq_len_min,
    seq_len_max,
    display_step=10,
    training_iters=1e8,
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
      seq_len=2*random.randint(seq_len_min, seq_len_max/2),
      nbits=n_input,
    )
    sess.run(
      model['optimizer'],
      feed_dict={
        model['x']: xs,
        model['y']: ys,
        model['steps']: nsteps,
      },
    )
    if step % display_step == 0:
        _, loss = sess.run(
          [model['pred'], model['cost']],
          feed_dict={
            model['x']: xs,
            model['y']: ys,
            model['steps']: nsteps,
          },
        )
        print "Iter " + str(step*batch_size) + (
          ", Minibatch Loss= " + "{:.6f}".format(loss)
        )
    if step % 1000 == 0:
      save_path = saver.save(sess, "checkpoints/model" + str(step) + ".ckpt")
      print "Model saved in file: %s" % save_path

    step += 1

  print "Optimization Finished!"

# Training Parameters
max_steps = 10
seq_len_min = 1
seq_len_max = 8
batch_size = 1
n_input = 3
mem_nrow = 50
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
