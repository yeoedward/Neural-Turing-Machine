import tensorflow as tf
from tensorflow.models.rnn import rnn_cell
from tensorflow.python.framework import ops
import tensorflow.python.user_ops as rotate
# This is unnecessary after rebuilding tensorflow with the
# user op in the appropriate directory.
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
  def __init__(
      self,
      n_inputs,
      n_outputs,
      n_hidden,
      mem_nrows,
      mem_ncols,
      n_heads,
    ):
    self.mem_nrows = mem_nrows
    self.mem_ncols = mem_ncols
    self.n_inputs = n_inputs
    self.n_outputs = n_outputs
    self.n_hidden = n_hidden
    self.n_heads = n_heads

  @property
  def state_size(self):
    return self.mem_nrows * self.mem_ncols + 2 * self.n_heads * self.mem_nrows

  @property
  def output_size(self):
    return self.n_outputs

  def zero_state(self, batch_size, dtype):
    M_bias = tf.random_uniform(
      tf.pack([batch_size, self.mem_nrows * self.mem_ncols]),
      0, 0.1,
    )
    indices = tf.zeros(tf.expand_dims(batch_size, 0), dtype=tf.int64)
    weight_biases = [
      tf.one_hot(indices, self.mem_nrows)
      for _ in xrange(2*self.n_heads)
    ]
    state_biases = tf.concat(1, [M_bias] + weight_biases)
    return state_biases

  @staticmethod
  def var_name(v, i, is_write):
    prefix = "write_" if is_write else "read_"
    return prefix + v + str(i)

  def add_head_params(self, weights, biases, i, init_min, init_max, is_write):
      key_name = NTMCell.var_name("key", i, is_write)
      weights[key_name] = tf.get_variable(
        name=key_name + "_weight",
        shape=[self.n_hidden, self.mem_ncols],
        initializer=tf.random_uniform_initializer(init_min, init_max),
      )
      biases[key_name] = tf.get_variable(
        name=key_name + "_bias",
        shape=[self.mem_ncols],
        initializer=tf.constant_initializer(0),
      )

      key_str_name = NTMCell.var_name("key_str", i, is_write)
      weights[key_str_name] = tf.get_variable(
        name=key_str_name + "_weight",
        shape=[self.n_hidden, 1],
        initializer=tf.random_uniform_initializer(init_min, init_max),
      )
      biases[key_str_name] = tf.get_variable(
        name=key_str_name + "_bias",
        shape=[1],
        initializer=tf.constant_initializer(0),
      )

      interp_name = NTMCell.var_name("interp", i, is_write)
      weights[interp_name] = tf.get_variable(
        name=interp_name + "_weight",
        shape=[self.n_hidden, 1],
        initializer=tf.random_uniform_initializer(init_min, init_max),
      )
      biases[interp_name] = tf.get_variable(
        name=interp_name + "_bias",
        shape=[1],
        initializer=tf.constant_initializer(0),
      )

      shift_name = NTMCell.var_name("shift", i, is_write)
      weights[shift_name] = tf.get_variable(
        name=shift_name + "_weight",
        shape=[self.n_hidden, 3],
        initializer=tf.random_uniform_initializer(init_min, init_max),
      )
      biases[shift_name] = tf.get_variable(
        name=shift_name + "_bias",
        shape=[3],
        initializer=tf.constant_initializer(0),
      )

      sharp_name = NTMCell.var_name("sharp", i, is_write)
      weights[sharp_name] =  tf.get_variable(
        name=sharp_name + "_weight",
        shape=[self.n_hidden, 1],
        initializer=tf.random_uniform_initializer(-0.1, 0.1),
      )
      biases[sharp_name] = tf.get_variable(
        name=sharp_name + "_bias",
        shape=[1],
        initializer=tf.constant_initializer(0),
      )

      if is_write:
        add_name = NTMCell.var_name("add", i, is_write)
        weights[add_name] = tf.get_variable(
          name=add_name + "_weight",
          shape=[self.n_hidden, self.mem_ncols],
          initializer=tf.random_uniform_initializer(init_min, init_max),
        )
        biases[add_name] = tf.get_variable(
          name=add_name + "_bias",
          shape=[self.mem_ncols],
          initializer=tf.constant_initializer(0),
        )

        erase_name = NTMCell.var_name("erase", i, is_write)
        weights[erase_name] = tf.get_variable(
          name=erase_name + "_weight",
          shape=[self.n_hidden, self.mem_ncols],
          initializer=tf.random_uniform_initializer(init_min, init_max),
        )
        biases[erase_name] = tf.get_variable(
          name=erase_name + "_bias",
          shape=[self.mem_ncols],
          initializer=tf.constant_initializer(0),
        )

  def get_params(self):
    n_first_layer = self.n_inputs + self.n_heads * self.mem_ncols
    init_min = -1
    init_max = 1
    weights = {
      "hidden": tf.get_variable(
        name="hidden_weight",
        shape=[n_first_layer, self.n_hidden],
        initializer=tf.random_uniform_initializer(init_min, init_max),
      ),
      "output": tf.get_variable(
        name="output_weight",
        shape=[self.n_hidden, self.n_outputs],
        initializer=tf.random_uniform_initializer(init_min, init_max),
      ),
    }
    biases = {
      "hidden": tf.get_variable(
        name="hidden_bias",
        shape=[self.n_hidden],
        initializer=tf.constant_initializer(0),
      ),
      "output": tf.get_variable(
        name="output_bias",
        shape=[self.n_outputs],
        initializer=tf.constant_initializer(0),
      ),
    }

    for i in xrange(self.n_heads):
      self.add_head_params(
        weights=weights,
        biases=biases,
        i=i,
        init_min=init_min,
        init_max=init_max,
        is_write=True,
      )
      self.add_head_params(
        weights=weights,
        biases=biases,
        i=i,
        init_min=init_min,
        init_max=init_max,
        is_write=False,
      )

    return weights, biases

  @staticmethod
  def head_outputs(weights, biases, hidden, i, is_write):
    key_name = NTMCell.var_name("key", i, is_write)
    key = tf.matmul(hidden, weights[key_name]) + biases[key_name]
    key = tf.nn.softplus(key)

    key_str_name = NTMCell.var_name("key_str", i, is_write)
    key_str = tf.matmul(hidden, weights[key_str_name]) + biases[key_str_name]
    key_str = tf.nn.relu(key_str)

    interp_name = NTMCell.var_name("interp", i, is_write)
    interp = tf.matmul(hidden, weights[interp_name]) + biases[interp_name]
    interp = tf.sigmoid(interp)

    shift_name = NTMCell.var_name("shift", i, is_write)
    shift = tf.matmul(hidden, weights[shift_name]) + biases[shift_name]
    shift = tf.nn.softmax(shift)

    sharp_name = NTMCell.var_name("sharp", i, is_write)
    sharp = tf.matmul(hidden, weights[sharp_name]) + biases[sharp_name]
    sharp = tf.nn.relu(sharp) + 1

    head = {
      "key": key,
      "key_str": key_str,
      "interp": interp,
      "shift": shift,
      "sharp": sharp,
    }

    if is_write:
      add_name = NTMCell.var_name("add", i, is_write)
      add = tf.matmul(hidden, weights[add_name]) + biases[add_name]
      add = tf.nn.relu(add)

      erase_name = NTMCell.var_name("erase", i, is_write)
      erase = tf.matmul(hidden, weights[erase_name]) + biases[erase_name]
      erase = tf.sigmoid(erase)
      
      head["add"] = add
      head["erase"] = erase

    return head

  def controller(self, inputs, reads):
    weights, biases = self.get_params()
    first_layer = tf.concat(1, [inputs] + reads)
    #first_layer = tf.Print(first_layer, [first_layer], "first_layer", summarize=20)
    hidden = tf.matmul(first_layer, weights["hidden"]) + biases["hidden"]
    hidden = tf.nn.relu(hidden)

    output = tf.matmul(hidden, weights["output"]) + biases["output"]
    
    write_heads = []
    read_heads = []
    for i in xrange(self.n_heads):
      write_heads.append(
        NTMCell.head_outputs(weights, biases, hidden, i, is_write=True))
      read_heads.append(
        NTMCell.head_outputs(weights, biases, hidden, i, is_write=False))

    return output, write_heads, read_heads

  @staticmethod
  def magnitude(tens, dim):
    return tf.sqrt(tf.reduce_sum(tf.square(tens), dim))

  @staticmethod
  def address(M0, w0, head):
    # Content focusing
    # Compute cosine similarity
    key = tf.expand_dims(head["key"], 1)
    key_matches = tf.batch_matmul(key, tf.transpose(M0, [0, 2, 1]))
    key_matches = tf.squeeze(key_matches)
    key_mag = tf.expand_dims(NTMCell.magnitude(head["key"], 1), 1)
    M_col_mag = NTMCell.magnitude(M0, 2)
    cosine_sim = key_matches / (key_mag * M_col_mag)
    # Compute content weights
    wc = tf.nn.softmax(head["key_str"] * cosine_sim)

    # Location focusing
    wg = head["interp"] * wc + (1 - head["interp"]) * w0
    ws = rotate.ntm_rotate(wg, head["shift"])
    ws_pow = tf.pow(ws, head["sharp"])

    #ws_pow = tf.Print(ws_pow, [w0], "w0", summarize=1000)
    #ws_pow = tf.Print(ws_pow, [key], "key", summarize=1000)
    #ws_pow = tf.Print(ws_pow, [head["key_str"]], "key_str", summarize=1000)
    #ws_pow = tf.Print(ws_pow, [wc], "wc", summarize=1000)
    #ws_pow = tf.Print(ws_pow, [head["interp"]], "interp", summarize=1000)
    #ws_pow = tf.Print(ws_pow, [head["shift"]], "shift", summarize=1000)
    #ws_pow = tf.Print(ws_pow, [wg], "wg", summarize=1000)
    #ws_pow = tf.Print(ws_pow, [head["sharp"]], "sharp", summarize=1000)
    #ws_pow = tf.Print(ws_pow, [ws_pow], "ws_pow", summarize=1000)

    w1 = ws_pow / tf.reduce_sum(ws_pow, 1, keep_dims=True)


    return w1

  def deserialize(self, state):
    # Deserialize state from previous timestep.
    M0 = tf.slice(
      state,
      [0, 0],
      [-1, self.mem_nrows * self.mem_ncols],
    )
    M0 = tf.reshape(M0, [-1, self.mem_nrows, self.mem_ncols])

    state_idx = self.mem_nrows * self.mem_ncols

    # Deserialize read weights from previous time step.
    read_w0s = []
    for i in xrange(self.n_heads):
      # Number of weights == Rows of memory matrix
      w0 = tf.slice(state, [0, state_idx], [-1, self.mem_nrows])
      read_w0s.append(w0)
      state_idx += self.mem_nrows

    # Do the same for write heads.
    write_w0s = []
    for _ in xrange(self.n_heads):
      w0 = tf.slice(state, [0, state_idx], [-1, self.mem_nrows])
      write_w0s.append(w0)
      state_idx += self.mem_nrows

    tf.Assert(
      tf.equal(state_idx, tf.shape(state)[1]),
      [tf.shape(state)],
    )

    return M0, write_w0s, read_w0s

  def serialize(self, M1, write_w1s, read_w1s):
    # Serialize state for next timestep
    s_read_w1s = []
    for i in xrange(len(read_w1s)):
      w1 = read_w1s[i]
      s_read_w1s.append(
        tf.reshape(w1, [-1, self.mem_nrows]),
      )
    s_write_w1s = []
    for i in xrange(len(write_w1s)):
      w1 = write_w1s[i]
      s_write_w1s.append(
        tf.reshape(w1, [-1, self.mem_nrows]),
      )
    sM1 = tf.reshape(M1, [-1, self.mem_nrows * self.mem_ncols])
    new_state = tf.concat(1, [sM1] + s_read_w1s + s_write_w1s)

    return new_state

  def read(self, M0, read_w0s):
    reads = []
    for i in xrange(self.n_heads):
      w0 = read_w0s[i]
      r = tf.batch_matmul(tf.expand_dims(w0, 1), M0)
      r = tf.squeeze(r, [1])
      reads.append(r)
    return reads

  def write(self, M0, write_w0s, write_heads):
    write_w1s = []
    for i in xrange(self.n_heads):
      head = write_heads[i]
      w0 = write_w0s[i]
      w1 = NTMCell.address(M0, w0, head)
      write_w1s.append(w1)

    M1 = M0
    # Erases
    for w1 in write_w1s:
      we = 1 - tf.batch_matmul(
        tf.expand_dims(w1, 2),
        tf.expand_dims(head["erase"], 1)
      )  
      M1 = M1 * we

    # Writes
    for w1 in write_w1s:
      add = tf.batch_matmul(
        tf.expand_dims(w1, 2),
        tf.expand_dims(head["add"], 1),
      )
      M1 = M1 + add

    return M1, write_w1s

  def read_weights(self, M0, read_w0s, read_heads):
    read_w1s = []
    # Compute read weights
    for i in xrange(self.n_heads): 
      head = read_heads[i]
      w0 = read_w0s[i]
      w1 = NTMCell.address(M0, w0, head)
      read_w1s.append(w1)
    return read_w1s

  def __call__(self, inputs, state, scope=None):
    with tf.variable_scope(scope or type(self).__name__):
      M0, write_w0s, read_w0s = self.deserialize(state)
      # Read from memory
      reads = self.read(M0, read_w0s)
      # Run inputs and read through controller.
      output, write_heads, read_heads = self.controller(inputs, reads)
      # Compute new read weights
      read_w1s = self.read_weights(M0, read_w0s, read_heads)
      # Perform Writes
      M1, write_w1s = self.write(M0, write_w0s, write_heads)
      new_state = self.serialize(M1, write_w1s, read_w1s)
      return output, new_state
