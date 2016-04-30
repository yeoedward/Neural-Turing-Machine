15-453 Final Project

Implementation of a Neural Turing Machine (NTM) using Tensor Flow (v0.8)

Paper: http://arxiv.org/abs/1410.5401

TODO:

  Figure out why loss isn't decreasing to 0.
    content weight is very spread out.

  Run on GPUs.

Training details:

  Train in minibatches:

    Freshly generated, because we can.

    All sequences within minibatch are of the same random length.

      Tensors that are of the same shape are easier to deal with and
        possibly more efficient because of matrix ops.
    
    dynamic_rnn requires the number of steps to be the same across minibatches
      so we need to pad to max_steps.

Build user op in tensorflow/tensorflow/core/user_ops/:

bazel build -c opt //tensorflow/core/user_ops:rotate.so

Upgrade to latest Tensorflow v0.8 (Pip automatically uses v0.7+):

sudo pip install --upgrade https://storage.googleapis.com/tensorflow/mac/tensorflow-0.8.0-py2-none-any.whl

Citations:

Some RNN Code was adapted from:

https://github.com/aymericdamien/TensorFlow-Examples/tree/master/examples/3%20-%20Neural%20Networks
