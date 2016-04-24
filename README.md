15-453 Final Project

Implementation of a Neural Turing Machine (NTM) using Tensor Flow

Paper: http://arxiv.org/abs/1410.5401

TODO:

  Implement batched training:

    Replace split with unpack()

  Implement NTM:

    See RNN for reference about how to implement istate.

Copy Task:

  Train in minibatches:

    Freshly generated because we can.

    All sequences within minibatch are of the same random length.

      Tensors that are of the same shape are easier to deal with and
        possibly more efficient because of matrix ops.
    
    For RNN, we set max_steps so we don't have to create multiple computation
      graphs. In particular, the tf.split() function requires the number of
      steps as a python integer.

Citations:

Some RNN Code was adapted from:

https://github.com/aymericdamien/TensorFlow-Examples/tree/master/examples/3%20-%20Neural%20Networks
