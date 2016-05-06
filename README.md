# 15-453 Final Project

Implementation of a Neural Turing Machine (NTM) using Tensor Flow (v0.8)

Original Paper: http://arxiv.org/abs/1410.5401

## Installation

  1. Clone TensorFlow (v0.8) into the root folder of this project.
  You can find a link to the github repo at https://www.tensorflow.org/. Follow the instructions to
  build and install it. You should build from source because we use a custom user op implemented
  in C++.

  2. In the folder `rotate_op/` you will find three files: `BUILD`, `rotate.cc` and `rotate_grad.cc`.
  These are user operations that we wrote. Copy them into `tensorflow/tensorflow/core/user_ops`.
  Then run the following command `bazel build -c opt //tensorflow/core/user_ops:rotate.so` in that directory.
  This generates the `.so` file that will be loaded by `ntm.py`. If you did not clone tensorflow
  into the root folder of this project as instructed, please modify the path in `ntm.py` accordingly.
  
  3. To execute the copy task, run `python copy_task.py`.
  
## Checkpoints

Checkpoint files are automatically saved by `copy_task.py` every 1000 training iterations.
You can reload a saved model and run it with custom inputs using `analyze.py`.
You might have to make some changes specific to your experiment.

## Report

You will find the accompanying final report in the `report/` folder. It uses some images from the
`images/` folder. Print statements along with `parse.py` were used to log
some of the internal state of the NTM, like the read and write head positions that we show in the
report.

## Code Structure

You will find all the NTM specific code in `ntm.py`. The boilerplate/experiment-harness code for
the copy task can be found in `copy_task.py`. Files prefixed with `test_` are small test scripts.
Most of them just check if the code runs without crashing. The experiment code
for the context-free parenthesis language experiment can be found in `dyck_task.py`.
Unfortunately due to lack of time we were unable to write it up.

## Resources:

https://www.tensorflow.org/

https://medium.com/snips-ai/ntm-lasagne-a-library-for-neural-turing-machines-in-lasagne-2cdce6837315#.17cngz3vj

http://awawfumin.blogspot.com/2015/03/neural-turing-machines-implementation.html

https://blog.wtf.sg/2015/01/15/neural-turing-machines-faq/#more-843
