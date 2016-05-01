15-453 Final Project

Implementation of a Neural Turing Machine (NTM) using Tensor Flow (v0.8)

Paper: http://arxiv.org/abs/1410.5401

TODO:

  Tune params.

  Write report.

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

Saved models:

  We copy the source code too to ensure that the checkpoint file is
    compatible with the model. There is probably (hopefully) a better way
    to do this.

Resources:

https://www.tensorflow.org/

https://medium.com/snips-ai/ntm-lasagne-a-library-for-neural-turing-machines-in-lasagne-2cdce6837315#.17cngz3vj

http://awawfumin.blogspot.com/2015/03/neural-turing-machines-implementation.html

https://blog.wtf.sg/2015/01/15/neural-turing-machines-faq/#more-843
