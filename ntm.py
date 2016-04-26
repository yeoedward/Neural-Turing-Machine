import tensorflow as tf
from tensorflow.models.rnn import rnn_cell
# WARNING: This is not part of the v0.8 public API and might break 
# in future versions of tensorflow.
from tensorflow.python.ops import control_flow_ops, tensor_array_ops

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
        initializer=tf.random_normal_initializer(0, 0.1),
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
    output = tf.sigmoid(output)

    key = tf.matmul(hidden, weights["key"]) + biases["key"]

    key_str = tf.matmul(hidden, weights["key_str"]) + biases["key_str"]
    key_str = tf.exp(key_str) + 1

    interp = tf.matmul(hidden, weights["interp"]) + biases["interp"]
    interp = tf.sigmoid(interp)

    shift = tf.matmul(hidden, weights["shift"]) + biases["shift"]
    shift = tf.exp(shift) / tf.reduce_sum(tf.exp(shift))

    sharp = tf.matmul(hidden, weights["sharp"]) + biases["sharp"]
    sharp = tf.exp(sharp) + 2

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

  @staticmethod
  def conv_sum(wg_i, shift_i, row_idx):
    i = tf.constant(0)
    total = tf.constant(0.)
    def conv_step(i, wg_i, shift_i, row_idx, total): 
      shift_len = tf.shape(shift_i)[0]
      si = (row_idx - i) % shift_len
      si = tf.expand_dims((si + shift_len) % shift_len, 0)
      total += (tf.slice(wg_i, tf.expand_dims(i, 0), [1]) *
        tf.slice(shift_i, si, [1]))
      i += 1  
      return i, wg_i, shift_i, row_idx, total
    (_, _, _, _, total) = control_flow_ops.while_loop(
      cond=lambda i, wg_i, _1, _2, _3: i < tf.shape(wg_i)[0],
      body=conv_step,
      loop_vars=(i, wg_i, shift_i, row_idx, total),
    )
    return total

  @staticmethod
  def rotate(wg, shift):
    # WARNING: Not in public API.
    # Might break for versions after v0.8.
    ws = tensor_array_ops.TensorArray(
      dtype=wg.dtype,
      size=tf.reduce_prod(tf.shape(wg)),
    )
    def rotate_step(i, wg, shift, ws):
      batch_idx = i / tf.shape(wg)[1]
      row_idx = i % tf.shape(wg)[1]
      wg_i = tf.squeeze(tf.slice(wg, tf.pack([batch_idx, 0]), [1, -1]))
      shift_i = tf.squeeze(tf.slice(shift, tf.pack([batch_idx, 0]), [1, -1]))
      ws_i = NTMCell.conv_sum(wg_i, shift_i, row_idx)
      ws = ws.write(i, ws_i)
      i += 1
      return i, wg, shift, ws
    i = tf.constant(0)
    (_, _, _, ws) = control_flow_ops.while_loop(
      cond=lambda i, wg, _1, _2: i < tf.reduce_prod(tf.shape(wg)),
      body=rotate_step,
      loop_vars=(i, wg, shift, ws),
    )
    ws = tf.reshape(ws.pack(), tf.pack([tf.shape(wg)[0], tf.shape(wg)[1]]))
    return ws

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
      ws = NTMCell.rotate(wg, head["shift"])
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
