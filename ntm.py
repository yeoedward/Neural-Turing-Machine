import tensorflow as tf
from tensorflow.models.rnn import rnn_cell
from tensorflow.python.framework import ops
rotate = tf.load_op_library(
  "tensorflow/bazel-bin/tensorflow/core/user_ops/rotate.so",
)
@ops.RegisterGradient("NTMRotate")
def _ntm_rotate_grad(op, grad):
  weights = op.inputs[0]
  shifts = op.inputs[1]
  weights_grad, shifts_grad = rotate.ntm_rotate_grad(grad, weights, shifts);
  return [weights_grad, shifts_grad]

class NTMCell(rnn_cell.RNNCell):
  def __init__(self, n_inputs, n_hidden, mem_nrows, mem_ncols):
    self.mem_nrows = mem_nrows
    self.mem_ncols = mem_ncols
    self.n_inputs = n_inputs
    self.n_hidden = n_hidden

  @property
  def state_size(self):
    return self.mem_nrows * self.mem_ncols + self.mem_nrows

  @property
  def output_size(self):
    return self.n_inputs

  def zero_state(self, batch_size, dtype):
    state_size = self.mem_nrows * self.mem_ncols + self.mem_nrows
    return tf.ones([batch_size, state_size], dtype)

  def get_params(self):
  #TODO Expose constants as args in constructor.
    n_first_layer = self.n_inputs + self.mem_ncols
    weights = {
      "hidden": tf.get_variable(
        name="hidden_weight",
        shape=[n_first_layer, self.n_hidden],
        initializer=tf.random_normal_initializer(0, 0.1),
      ),
      "output": tf.get_variable(
        name="output_weight",
        shape=[self.n_hidden, self.n_inputs],
        initializer=tf.random_normal_initializer(0, 0.1),
      ),
      "key": tf.get_variable(
        name="key_weight",
        shape=[self.n_hidden, self.mem_ncols],
        initializer=tf.random_normal_initializer(0, 0.1),
      ),
      "key_str": tf.get_variable(
        name="key_str_weight",
        shape=[self.n_hidden, 1],
        initializer=tf.random_normal_initializer(0, 0.1),
      ),
      "interp": tf.get_variable(
        name="interp_weight",
        shape=[self.n_hidden, 1],
        initializer=tf.random_normal_initializer(0, 0.1),
      ),
      "shift": tf.get_variable(
        name="shift_weight",
        shape=[self.n_hidden, 3],
        initializer=tf.random_normal_initializer(0, 0.1),
      ),
      "sharp": tf.get_variable(
        name="sharp_weight",
        shape=[self.n_hidden, 1],
        initializer=tf.random_normal_initializer(0, 0.1),
      ),
      "add": tf.get_variable(
        name="add_weight",
        shape=[self.n_hidden, self.mem_ncols],
        initializer=tf.random_normal_initializer(0, 0.1),
      ),
      "erase": tf.get_variable(
        name="erase_weight",
        shape=[self.n_hidden, self.mem_ncols],
        initializer=tf.random_normal_initializer(0, 0.1),
      ),
    }

    biases = {
      "hidden": tf.get_variable(
        name="hidden_bias",
        shape=[self.n_hidden],
        initializer=tf.random_normal_initializer(1, 0.1),
      ),
      "output": tf.get_variable(
        name="output_bias",
        shape=[self.n_inputs],
        initializer=tf.random_normal_initializer(0, 0.1),
      ),
      "key": tf.get_variable(
        name="key_bias",
        shape=[self.mem_ncols],
        initializer=tf.random_normal_initializer(0, 0.1),
      ),
      "key_str": tf.get_variable(
        name="key_str_bias",
        shape=[1],
        initializer=tf.random_normal_initializer(0, 0.1),
      ),
      "interp": tf.get_variable(
        name="interp_bias",
        shape=[1],
        initializer=tf.random_normal_initializer(0, 0.1),
      ),
      "shift": tf.get_variable(
        name="shift_bias",
        shape=[3],
        initializer=tf.random_normal_initializer(0, 0.1),
      ),
      "sharp": tf.get_variable(
        name="sharp_bias",
        shape=[1],
        initializer=tf.random_normal_initializer(0, 0.1),
      ),
      "add": tf.get_variable(
        name="add_bias",
        shape=[self.mem_ncols],
        initializer=tf.random_normal_initializer(0, 0.1),
      ),
      "erase": tf.get_variable(
        name="erase_bias",
        shape=[self.mem_ncols],
        initializer=tf.random_normal_initializer(0, 0.1),
      ),
    }

    return weights, biases

  def controller(self, inputs, read):
    first_layer = tf.concat(1, [inputs, read])
    weights, biases = self.get_params()

    hidden = tf.matmul(first_layer, weights["hidden"]) + biases["hidden"]
    hidden = tf.nn.relu(hidden)

    output = tf.matmul(hidden, weights["output"]) + biases["output"]

    key = tf.matmul(hidden, weights["key"]) + biases["key"]

    key_str = tf.matmul(hidden, weights["key_str"]) + biases["key_str"]
    key_str = tf.exp(key_str)

    interp = tf.matmul(hidden, weights["interp"]) + biases["interp"]
    interp = tf.sigmoid(interp)

    shift = tf.matmul(hidden, weights["shift"]) + biases["shift"]
    shift = tf.exp(shift) / tf.reduce_sum(tf.exp(shift))

    sharp = tf.matmul(hidden, weights["sharp"]) + biases["sharp"]
    sharp = tf.exp(sharp) + 1

    add = tf.matmul(hidden, weights["add"]) + biases["add"]

    erase = tf.matmul(hidden, weights["erase"]) + biases["erase"]
    erase = tf.sigmoid(erase)

    return output, {
      "key": key,
      "key_str": key_str,
      "interp": interp,
      "shift": shift,
      "sharp": sharp,
      "add": add,
      "erase": erase,
    }

  @staticmethod
  def magnitude(tens, dim):
    return tf.sqrt(tf.reduce_sum(tf.square(tens), dim))

  def __call__(self, inputs, state, scope=None):
    with tf.variable_scope(scope or type(self).__name__):
      # Number of weights == Rows of memory matrix
      w0 = tf.slice(state, [0, 0], [-1, self.mem_nrows])
      M0 = tf.slice(state, [0, self.mem_nrows], [-1, -1])
      M0 = tf.reshape(M0, [-1, self.mem_nrows, self.mem_ncols])

      # Read
      read = tf.squeeze(tf.batch_matmul(tf.expand_dims(w0, 1), M0))

      # Run inputs and read through controller.
      output, head = self.controller(inputs, read)

      # Content focusing
      key = tf.expand_dims(head["key"], 1)
      key_matches = tf.batch_matmul(key, tf.transpose(M0, [0, 2, 1]))
      key_matches = tf.squeeze(key_matches)
      key_mag = tf.expand_dims(NTMCell.magnitude(head["key"], 1), 1)
      M_col_mag = NTMCell.magnitude(M0, 2)
      cosine_sim = key_matches / (key_mag * M_col_mag)
      amp_sim = tf.exp(head["key_str"] * cosine_sim) 
      wc = amp_sim / tf.reduce_sum(amp_sim, 1, keep_dims=True)

      # Location focusing
      wg = head["interp"] * wc + (1 - head["interp"]) * w0
      ws = rotate.ntm_rotate(wg, head["shift"])
      ws_pow = tf.pow(ws, head["sharp"])
      w1 = ws_pow / tf.reduce_sum(ws_pow)

      # Write
      we = 1 - tf.batch_matmul(
        tf.expand_dims(w1, 2),
        tf.expand_dims(head["erase"], 1)
      )  
      Me = M0 * we
      M1 = Me + tf.batch_matmul(
        tf.expand_dims(w1, 2),
        tf.expand_dims(head["add"], 1),
      )
     
      sw1 = tf.reshape(w1, [-1, self.mem_nrows])
      sM1 = tf.reshape(M1, [-1, self.mem_nrows * self.mem_ncols])
      new_state = tf.concat(1, [sw1, sM1])

      return output, new_state
