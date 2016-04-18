import tensorflow as tf
from tensorflow.models.rnn import rnn, rnn_cell
import random
import numpy as np

random.seed(1337)
NBITS = 8
MAX_SEQ_LENGTH = 20
def gen_seq(_nseqs):
    zeros = [0] * NBITS
    ones = [1] * NBITS
    xs = []
    ys = []
    for _ in xrange(_nseqs):
        seq_len = 20 # TODO random.randint(1, MAX_SEQ_LENGTH)
        # Note that we reserve the all-zero and all-one vectors as the 
        # padding and delimeter symbols
        seq = [random.randint(1, 2**NBITS - 1) for _ in xrange(seq_len)]
        # Convert each int to binary
        seq = [
                [int(digit) for digit in '{0:08b}'.format(num)]
                for num in seq
              ]
        # Dummy inputs after the delimeter / outputs before the delieter
        dummies = [zeros] * seq_len
        x = seq + [ones] + dummies
        y = dummies + [ones] + seq
        xs.append(x)
        ys.append(y)

    return np.array(xs), np.array(ys)

# Parameters
learning_rate = 0.001
training_iters = 100000
batch_size = 128
display_step = 10

# Network Parameters
n_input = NBITS
n_hidden = 128 # hidden layer num of features
# As we need tensors which are "cubic"
# In reality, our sequences are of different lengths i.e. jagged
max_steps = 2*MAX_SEQ_LENGTH + 1

# Batch size, nsteps, ninput
x = tf.placeholder("float", [None, None, n_input])
istate = tf.placeholder("float", [None, 2*n_hidden]) #state & cell => 2x n_hidden
y = tf.placeholder("float", [None, None, n_input])

weights = {
    'hidden': tf.Variable(tf.random_normal([n_input, n_hidden])),
    'out': tf.Variable(tf.random_normal([n_hidden, n_input]))
}
biases = {
    'hidden': tf.Variable(tf.random_normal([n_hidden])),
    'out': tf.Variable(tf.random_normal([n_input]))
}

def RNN(_X, _istate, _weights, _biases):
    # input shape: (batch_size, n_steps, n_input)
    _X = tf.transpose(_X, [1, 0, 2])  # permute n_steps and batch_size
    # Reshape to prepare input to hidden activation
    _X = tf.reshape(_X, [-1, n_input]) # (n_steps*batch_size, n_input)
    # Linear activation
    _X = tf.matmul(_X, _weights['hidden']) + _biases['hidden']

    # Define a lstm cell with tensorflow
    lstm_cell = rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)

    # Split data because rnn cell needs a list of inputs for the RNN inner loop
    _X = tf.split(0, max_steps, _X) # n_steps * (batch_size, n_hidden)

    # Get lstm cell output
                                
    # TODO: sequence_length=_timesteps
    # Figure out what to do with sequences of different lengths
    outputs, states = rnn.rnn(lstm_cell, _X,
                              initial_state=_istate)


    # Linear activation
    outputs = tf.pack(outputs)
    outputs = tf.transpose(outputs, [1, 0, 2])
    outputs = tf.reshape(outputs, [-1, n_hidden])
    preds = tf.matmul(outputs, _weights['out']) + _biases['out']
    return tf.reshape(preds, [-1, max_steps, n_input])


# Loss function that averages cross entropy, taking into account
# the variable lengths of the sequences.
#
# _pred: list of tensors, where each element at index t is
#        batch_size * outputs at timestep t
def var_seq_loss(_preds, _y):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(_preds, _y))

def predict(_preds):
    return tf.sigmoid(_preds)

pred = RNN(x, istate, weights, biases)

# Define loss and optimizer
cost = var_seq_loss(pred, y)
optimizer = tf.train.RMSPropOptimizer(
  learning_rate=learning_rate).minimize(cost)

# Initializing the variables
init = tf.initialize_all_variables()

# Generate training data
ndata = 128*10
xs, ys = gen_seq(ndata)
batch_idx = 0

# Launch the graph
sess = tf.Session()
sess.run(init)
step = 1
# Keep training until reach max iterations
while step * batch_size < training_iters:
    batch_xs = xs[batch_idx:batch_idx + batch_size]
    batch_ys = ys[batch_idx:batch_idx+batch_size]
    batch_idx = (batch_idx + batch_size) % ndata

    # Fit training using batch data
    sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys,
                                   istate: np.zeros((batch_size, 2*n_hidden))})
    if step % display_step == 0:
        # Calculate batch loss
        loss = sess.run(cost, feed_dict={
          x: batch_xs, y: batch_ys,
          istate: np.zeros((batch_size, 2*n_hidden))})
        print "Iter " + str(step*batch_size) + (
          ", Minibatch Loss= " + "{:.6f}".format(loss))
    step += 1
print "Optimization Finished!"

test_xs, test_ys = gen_seq(1)
print(test_xs.shape)
sess.run(predict(pred), feed_dict={x: test_xs,
                                   istate: np.zeros((1, 2*n_hidden))})
