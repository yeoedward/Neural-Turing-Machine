import tensorflow as tf
from tensorflow.models.rnn import rnn, rnn_cell
import numpy as np
import random

# Define loss and optimizer
def var_seq_loss(_preds, _y):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(_preds, _y))

def predict(_preds):
    return tf.sigmoid(_preds)

def create_rnn(max_steps, n_input, n_hidden):
  learning_rate = 0.001

  # Batch size, max_steps, n_input
  x = tf.placeholder("float", [None, None, n_input])
  istate = tf.placeholder("float", [None, 2*n_hidden])
  y = tf.placeholder("float", [None, None, n_input])

  weights = {
      'hidden': tf.Variable(tf.random_normal([n_input, n_hidden])),
      'out': tf.Variable(tf.random_normal([n_hidden, n_input]))
  }
  biases = {
      'hidden': tf.Variable(tf.random_normal([n_hidden])),
      'out': tf.Variable(tf.random_normal([n_input]))
  }

  X = x
  # Input: (batch_size, _max_steps, n_input)
  X = tf.transpose(X, [1, 0, 2]) 
  # (_max_steps * batch_size, n_input)
  X = tf.reshape(X, [-1, n_input])
  # Linear activation
  X = tf.matmul(X, weights['hidden']) + biases['hidden']
  # Define a lstm cell with tensorflow
  lstm_cell = rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)
  # Split data because rnn cell needs a list of inputs for the RNN inner loop
  # _max_steps * (batch_size, _n_hidden)
  X = tf.split(0, max_steps, X)
  outputs, states = rnn.rnn(lstm_cell, X,
                            initial_state=istate)
  outputs = tf.pack(outputs)

  # Transpose to (batch_size, _max_steps, _n_hidden)
  outputs = tf.transpose(outputs, [1, 0, 2])
  outputs = tf.reshape(outputs, [-1, n_hidden])
  preds = tf.matmul(outputs, weights['out']) + biases['out']
  pred = tf.reshape(preds, [-1, max_steps, n_input])

  # Loss functions
  cost = var_seq_loss(pred, y)
  optimizer = tf.train.RMSPropOptimizer(
  learning_rate=learning_rate).minimize(cost)

  return {
    'pred': pred,
    'cost': cost,
    'optimizer': optimizer,
    'x': x,
    'y': y,
    'istate': istate,
  }

def gen_seq(nseqs, max_steps, seq_len, nbits):
  assert 2 * seq_len + 1 <= max_steps
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
    npad = max_steps - (2 * seq_len + 1)
    pad = [zeros] * npad
    # Dummy inputs after the delimeter / outputs before the delimiter
    dummies = [zeros] * seq_len
    x = seq + [ones] + dummies + pad
    y = dummies + [ones] + seq + pad
    xs.append(x)
    ys.append(y)
  return np.array(xs), np.array(ys)

def train(
    model,
    n_input,
    n_hidden,
    max_steps,
    training_iters=100000,
    batch_size=128,
    display_step=10,
    ):
  # Initializing the variables
  init = tf.initialize_all_variables()

  # Launch the graph
  sess = tf.Session()
  sess.run(init)
  step = 1

  while step * batch_size < training_iters:
    seq_len = random.randint(1, 20)
    xs, ys = gen_seq(
      nseqs=batch_size,
      max_steps=max_steps,
      seq_len=seq_len,
      nbits=n_input,
    )
    sess.run(model['optimizer'],
             feed_dict={model['x']: xs, model['y']: ys,
                        model['istate']: np.zeros((batch_size, 2*n_hidden))})
    if step % display_step == 0:
        loss = sess.run(model['cost'], feed_dict={
          model['x']: xs, model['y']: ys,
          model['istate']: np.zeros((batch_size, 2*n_hidden))})
        print "Iter " + str(step*batch_size) + (
          ", Minibatch Loss= " + "{:.6f}".format(loss))
    step += 1
  print "Optimization Finished!"

def test(pred):
  test_xs, test_ys = gen_seq(1)
  print(test_xs.shape)
  sess.run(predict(pred), feed_dict={x: test_xs,
                                     istate: np.zeros((1, 2*n_hidden))})

# Training Parameters
max_steps = 41
n_input = 8
n_hidden = 128
model = create_rnn(max_steps, n_input, n_hidden)

if __name__ == "__main__":
  train(
    model=model,
    n_input=n_input,
    n_hidden=n_hidden,
    max_steps=max_steps,
  )
