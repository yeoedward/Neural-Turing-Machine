<!-- This file is machine generated: DO NOT EDIT! -->

# Wraps python functions

Note: Functions taking `Tensor` arguments can also take anything accepted by
[`tf.convert_to_tensor`](framework.md#convert_to_tensor).

[TOC]

## Script Language Operators.

TensorFlow provides allows you to wrap python/numpy functions as
TensorFlow operators.

## Other Functions and Classes
- - -

### `tf.py_func(func, inp, Tout, name=None)` {#py_func}

Wraps a python function and uses it as a tensorflow op.

Given a python function `func`, which takes numpy arrays as its
inputs and returns numpy arrays as its outputs. E.g.,

  def my_func(x):
    return np.sinh(x)
  inp = tf.placeholder(..., tf.float32)
  y = py_func(my_func, [inp], [tf.float32])

The above snippet constructs a tf graph which invokes a numpy
sinh(x) as an op in the graph.

##### Args:


*  <b>`func`</b>: A python function.
*  <b>`inp`</b>: A list of `Tensor`.
*  <b>`Tout`</b>: A list of tensorflow data types indicating what `func`
        returns.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A list of `Tensor` which `func` computes.


