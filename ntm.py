import tensorflow as tf
from tensorflow.models.rnn import rnn_cell
# WARNING: This is not part of the public API and might break 
# in future versions of tensorflow.
from tensorflow.python.ops import control_flow_ops, tensor_array_ops

class NTMCell(rnn_cell.RNNCell):
  def __init__(self, mem_nrows, mem_ncols):
    self.mem_nrows = mem_nrows
    self.mem_ncols = mem_ncols

  @property
  def state_size(self):
    return self.mem_nrows * self.mem_ncols + self.mem_nrows

  @property
  def output_size(self):
    raise Exception("Not implemented yet")

  def controller(self, inputs, r):
    raise Exception("Not implemented yet")

  @staticmethod
  def magnitude(tens, dim):
    return tf.sqrt(tf.reduce_sum(tf.square(tens), dim))

  @staticmethod
  def conv_sum(wg_i, shift_i, row_idx):
    i = tf.constant(0)
    total = tf.constant(0)
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
      # Reshape for batch_matmul
      w0 = tf.expand_dims(w0, 1)
      M0 = tf.slice(state, [0, self.mem_nrows], [-1, -1])
      M0 = tf.reshape(M0, [-1, self.mem_nrows, self.mem_ncols])

      # Read
      read = tf.batch_matmul(w0, M0)

      # Run inputs and read through controller.
      output,
      key, key_str,
      interp, shift, sharp,
      add, erase = controller(inputs, read)

      # Content focusing
      key = tf.expand_dims(key, 1)
      key_matches = tf.batch_matmul(key, M0)
      key_mag = magnitude(key, 1)
      M_col_mag = magnitude(M, 1)
      cosine_sim = key_matches / (key_mag * M_col_mag)
      amp_sim = tf.exp(key_str * cosine_sim) 
      wc = amp_sim / tf.reduce_sum(amp_sim, 1)

      # Location focusing
      wg = interp * wc + (1 - interp) * w0
      ws = rotate(wg, shift)
      ws_pow = tf.pow(ws, sharp)
      w1 = ws_pow / tf.reduce_sum(ws_pow)

      # Write
      we = 1 - tf.batch_matmul(
        tf.expand_dims(w1, 2),
        tf.expand_dims(erase, 1)
      )  
      Me = M0 * we
      M1 = Me + tf.batch_matmul(
        tf.expand_dims(w1, 2),
        tf.expand_dims(add, 1),
      )
     
      sw1 = tf.reshape(w1, [-1])
      sM1 = tf.reshape(M1, [-1])
      new_state = tf.concat(0, [sw1, sM1])

      return output, new_state
