<!-- This file is machine generated: DO NOT EDIT! -->

# Variables

Note: Functions taking `Tensor` arguments can also take anything accepted by
[`tf.convert_to_tensor`](framework.md#convert_to_tensor).

[TOC]

## Variables

- - -

### `class tf.Variable` {#Variable}

See the [Variables How To](../../how_tos/variables/index.md) for a high
level overview.

A variable maintains state in the graph across calls to `run()`. You add a
variable to the graph by constructing an instance of the class `Variable`.

The `Variable()` constructor requires an initial value for the variable,
which can be a `Tensor` of any type and shape. The initial value defines the
type and shape of the variable. After construction, the type and shape of
the variable are fixed. The value can be changed using one of the assign
methods.

If you want to change the shape of a variable later you have to use an
`assign` Op with `validate_shape=False`.

Just like any `Tensor`, variables created with `Variable()` can be used as
inputs for other Ops in the graph. Additionally, all the operators
overloaded for the `Tensor` class are carried over to variables, so you can
also add nodes to the graph by just doing arithmetic on variables.

```python
import tensorflow as tf

# Create a variable.
w = tf.Variable(<initial-value>, name=<optional-name>)

# Use the variable in the graph like any Tensor.
y = tf.matmul(w, ...another variable or tensor...)

# The overloaded operators are available too.
z = tf.sigmoid(w + y)

# Assign a new value to the variable with `assign()` or a related method.
w.assign(w + 1.0)
w.assign_add(1.0)
```

When you launch the graph, variables have to be explicitly initialized before
you can run Ops that use their value. You can initialize a variable by
running its *initializer op*, restoring the variable from a save file, or
simply running an `assign` Op that assigns a value to the variable. In fact,
the variable *initializer op* is just an `assign` Op that assigns the
variable's initial value to the variable itself.

```python
# Launch the graph in a session.
with tf.Session() as sess:
    # Run the variable initializer.
    sess.run(w.initializer)
    # ...you now can run ops that use the value of 'w'...
```

The most common initialization pattern is to use the convenience function
`initialize_all_variables()` to add an Op to the graph that initializes
all the variables. You then run that Op after launching the graph.

```python
# Add an Op to initialize all variables.
init_op = tf.initialize_all_variables()

# Launch the graph in a session.
with tf.Session() as sess:
    # Run the Op that initializes all variables.
    sess.run(init_op)
    # ...you can now run any Op that uses variable values...
```

If you need to create a variable with an initial value dependent on another
variable, use the other variable's `initialized_value()`. This ensures that
variables are initialized in the right order.

All variables are automatically collected in the graph where they are
created. By default, the constructor adds the new variable to the graph
collection `GraphKeys.VARIABLES`. The convenience function
`all_variables()` returns the contents of that collection.

When building a machine learning model it is often convenient to distinguish
betwen variables holding the trainable model parameters and other variables
such as a `global step` variable used to count training steps. To make this
easier, the variable constructor supports a `trainable=<bool>` parameter. If
`True`, the new variable is also added to the graph collection
`GraphKeys.TRAINABLE_VARIABLES`. The convenience function
`trainable_variables()` returns the contents of this collection. The
various `Optimizer` classes use this collection as the default list of
variables to optimize.


Creating a variable.

- - -

#### `tf.Variable.__init__(initial_value=None, trainable=True, collections=None, validate_shape=True, caching_device=None, name=None, variable_def=None, dtype=None)` {#Variable.__init__}

Creates a new variable with value `initial_value`.

The new variable is added to the graph collections listed in `collections`,
which defaults to `[GraphKeys.VARIABLES]`.

If `trainable` is `True` the variable is also added to the graph collection
`GraphKeys.TRAINABLE_VARIABLES`.

This constructor creates both a `variable` Op and an `assign` Op to set the
variable to its initial value.

##### Args:


*  <b>`initial_value`</b>: A `Tensor`, or Python object convertible to a `Tensor`,
    which is the initial value for the Variable. The initial value must have
    a shape specified unless `validate_shape` is set to False. Can also be a
    callable with no argument that returns the initial value when called. In
    that case, `dtype` must be specified. (Note that initializer functions
    from init_ops.py must first be bound to a shape before being used here.)
*  <b>`trainable`</b>: If `True`, the default, also adds the variable to the graph
    collection `GraphKeys.TRAINABLE_VARIABLES`. This collection is used as
    the default list of variables to use by the `Optimizer` classes.
*  <b>`collections`</b>: List of graph collections keys. The new variable is added to
    these collections. Defaults to `[GraphKeys.VARIABLES]`.
*  <b>`validate_shape`</b>: If `False`, allows the variable to be initialized with a
    value of unknown shape. If `True`, the default, the shape of
    `initial_value` must be known.
*  <b>`caching_device`</b>: Optional device string describing where the Variable
    should be cached for reading.  Defaults to the Variable's device.
    If not `None`, caches on another device.  Typical use is to cache
    on the device where the Ops using the Variable reside, to deduplicate
    copying through `Switch` and other conditional statements.
*  <b>`name`</b>: Optional name for the variable. Defaults to `'Variable'` and gets
    uniquified automatically.
*  <b>`variable_def`</b>: `VariableDef` protocol buffer. If not `None`, recreates
    the Variable object with its contents. `variable_def` and the other
    arguments are mutually exclusive.
*  <b>`dtype`</b>: If set, initial_value will be converted to the given type.
    If `None`, either the datatype will be kept (if `initial_value` is
    a Tensor), or `convert_to_tensor` will decide.

##### Returns:

  A Variable.

##### Raises:


*  <b>`ValueError`</b>: If both `variable_def` and initial_value are specified.
*  <b>`ValueError`</b>: If the initial value is not specified, or does not have a
    shape and `validate_shape` is `True`.


- - -

#### `tf.Variable.initialized_value()` {#Variable.initialized_value}

Returns the value of the initialized variable.

You should use this instead of the variable itself to initialize another
variable with a value that depends on the value of this variable.

```python
# Initialize 'v' with a random tensor.
v = tf.Variable(tf.truncated_normal([10, 40]))
# Use `initialized_value` to guarantee that `v` has been
# initialized before its value is used to initialize `w`.
# The random values are picked only once.
w = tf.Variable(v.initialized_value() * 2.0)
```

##### Returns:

  A `Tensor` holding the value of this variable after its initializer
  has run.



Changing a variable value.

- - -

#### `tf.Variable.assign(value, use_locking=False)` {#Variable.assign}

Assigns a new value to the variable.

This is essentially a shortcut for `assign(self, value)`.

##### Args:


*  <b>`value`</b>: A `Tensor`. The new value for this variable.
*  <b>`use_locking`</b>: If `True`, use locking during the assignment.

##### Returns:

  A `Tensor` that will hold the new value of this variable after
  the assignment has completed.


- - -

#### `tf.Variable.assign_add(delta, use_locking=False)` {#Variable.assign_add}

Adds a value to this variable.

 This is essentially a shortcut for `assign_add(self, delta)`.

##### Args:


*  <b>`delta`</b>: A `Tensor`. The value to add to this variable.
*  <b>`use_locking`</b>: If `True`, use locking during the operation.

##### Returns:

  A `Tensor` that will hold the new value of this variable after
  the addition has completed.


- - -

#### `tf.Variable.assign_sub(delta, use_locking=False)` {#Variable.assign_sub}

Subtracts a value from this variable.

This is essentially a shortcut for `assign_sub(self, delta)`.

##### Args:


*  <b>`delta`</b>: A `Tensor`. The value to subtract from this variable.
*  <b>`use_locking`</b>: If `True`, use locking during the operation.

##### Returns:

  A `Tensor` that will hold the new value of this variable after
  the subtraction has completed.


- - -

#### `tf.Variable.scatter_sub(sparse_delta, use_locking=False)` {#Variable.scatter_sub}

Subtracts `IndexedSlices` from this variable.

This is essentially a shortcut for `scatter_sub(self, sparse_delta.indices,
sparse_delta.values)`.

##### Args:


*  <b>`sparse_delta`</b>: `IndexedSlices` to be subtracted from this variable.
*  <b>`use_locking`</b>: If `True`, use locking during the operation.

##### Returns:

  A `Tensor` that will hold the new value of this variable after
  the scattered subtraction has completed.

##### Raises:


*  <b>`ValueError`</b>: if `sparse_delta` is not an `IndexedSlices`.


- - -

#### `tf.Variable.count_up_to(limit)` {#Variable.count_up_to}

Increments this variable until it reaches `limit`.

When that Op is run it tries to increment the variable by `1`. If
incrementing the variable would bring it above `limit` then the Op raises
the exception `OutOfRangeError`.

If no error is raised, the Op outputs the value of the variable before
the increment.

This is essentially a shortcut for `count_up_to(self, limit)`.

##### Args:


*  <b>`limit`</b>: value at which incrementing the variable raises an error.

##### Returns:

  A `Tensor` that will hold the variable value before the increment. If no
  other Op modifies this variable, the values produced will all be
  distinct.



- - -

#### `tf.Variable.eval(session=None)` {#Variable.eval}

In a session, computes and returns the value of this variable.

This is not a graph construction method, it does not add ops to the graph.

This convenience method requires a session where the graph containing this
variable has been launched. If no session is passed, the default session is
used.  See the [Session class](../../api_docs/python/client.md#Session) for
more information on launching a graph and on sessions.

```python
v = tf.Variable([1, 2])
init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    # Usage passing the session explicitly.
    print(v.eval(sess))
    # Usage with the default session.  The 'with' block
    # above makes 'sess' the default session.
    print(v.eval())
```

##### Args:


*  <b>`session`</b>: The session to use to evaluate this variable. If
    none, the default session is used.

##### Returns:

  A numpy `ndarray` with a copy of the value of this variable.



Properties.

- - -

#### `tf.Variable.name` {#Variable.name}

The name of this variable.


- - -

#### `tf.Variable.dtype` {#Variable.dtype}

The `DType` of this variable.


- - -

#### `tf.Variable.get_shape()` {#Variable.get_shape}

The `TensorShape` of this variable.

##### Returns:

  A `TensorShape`.


- - -

#### `tf.Variable.device` {#Variable.device}

The device of this variable.


- - -

#### `tf.Variable.initializer` {#Variable.initializer}

The initializer operation for this variable.


- - -

#### `tf.Variable.graph` {#Variable.graph}

The `Graph` of this variable.


- - -

#### `tf.Variable.op` {#Variable.op}

The `Operation` of this variable.



#### Other Methods
- - -

#### `tf.Variable.from_proto(variable_def)` {#Variable.from_proto}

Returns a `Variable` object created from `variable_def`.


- - -

#### `tf.Variable.initial_value` {#Variable.initial_value}

Returns the Tensor used as the initial value for the variable.

Note that this is different from `initialized_value()` which runs
the op that initializes the variable before returning its value.
This method returns the tensor that is used by the op that initializes
the variable.

##### Returns:

  A `Tensor`.


- - -

#### `tf.Variable.ref()` {#Variable.ref}

Returns a reference to this variable.

You usually do not need to call this method as all ops that need a reference
to the variable call it automatically.

Returns is a `Tensor` which holds a reference to the variable.  You can
assign a new value to the variable by passing the tensor to an assign op.
See [`value()`](#Variable.value) if you want to get the value of the
variable.

##### Returns:

  A `Tensor` that is a reference to the variable.


- - -

#### `tf.Variable.to_proto()` {#Variable.to_proto}

Converts a `Variable` to a `VariableDef` protocol buffer.

##### Returns:

  A `VariableDef` protocol buffer.


- - -

#### `tf.Variable.value()` {#Variable.value}

Returns the last snapshot of this variable.

You usually do not need to call this method as all ops that need the value
of the variable call it automatically through a `convert_to_tensor()` call.

Returns a `Tensor` which holds the value of the variable.  You can not
assign a new value to this tensor as it is not a reference to the variable.
See [`ref()`](#Variable.ref) if you want to get a reference to the
variable.

To avoid copies, if the consumer of the returned value is on the same device
as the variable, this actually returns the live value of the variable, not
a copy.  Updates to the variable are seen by the consumer.  If the consumer
is on a different device it will get a copy of the variable.

##### Returns:

  A `Tensor` containing the value of the variable.




## Variable helper functions

TensorFlow provides a set of functions to help manage the set of variables
collected in the graph.

- - -

### `tf.all_variables()` {#all_variables}

Returns all variables that must be saved/restored.

The `Variable()` constructor automatically adds new variables to the graph
collection `GraphKeys.VARIABLES`. This convenience function returns the
contents of that collection.

##### Returns:

  A list of `Variable` objects.


- - -

### `tf.trainable_variables()` {#trainable_variables}

Returns all variables created with `trainable=True`.

When passed `trainable=True`, the `Variable()` constructor automatically
adds new variables to the graph collection
`GraphKeys.TRAINABLE_VARIABLES`. This convenience function returns the
contents of that collection.

##### Returns:

  A list of Variable objects.


- - -

### `tf.local_variables()` {#local_variables}

Returns all variables created with collection=[LOCAL_VARIABLES].

##### Returns:

  A list of local Variable objects.


- - -

### `tf.moving_average_variables()` {#moving_average_variables}

Returns all variables that maintain their moving averages.

If an `ExponentialMovingAverage` object is created and the `apply()`
method is called on a list of variables, these variables will
be added to the `GraphKeys.MOVING_AVERAGE_VARIABLES` collection.
This convenience function returns the contents of that collection.

##### Returns:

  A list of Variable objects.



- - -

### `tf.initialize_all_variables()` {#initialize_all_variables}

Returns an Op that initializes all variables.

This is just a shortcut for `initialize_variables(all_variables())`

##### Returns:

  An Op that initializes all variables in the graph.


- - -

### `tf.initialize_variables(var_list, name='init')` {#initialize_variables}

Returns an Op that initializes a list of variables.

After you launch the graph in a session, you can run the returned Op to
initialize all the variables in `var_list`. This Op runs all the
initializers of the variables in `var_list` in parallel.

Calling `initialize_variables()` is equivalent to passing the list of
initializers to `Group()`.

If `var_list` is empty, however, the function still returns an Op that can
be run. That Op just has no effect.

##### Args:


*  <b>`var_list`</b>: List of `Variable` objects to initialize.
*  <b>`name`</b>: Optional name for the returned operation.

##### Returns:

  An Op that run the initializers of all the specified variables.


- - -

### `tf.initialize_local_variables()` {#initialize_local_variables}

Returns an Op that initializes all local variables.

This is just a shortcut for `initialize_variables(local_variables())`

##### Returns:

  An Op that initializes all local variables in the graph.


- - -

### `tf.is_variable_initialized(variable)` {#is_variable_initialized}

Returns an Op to check if a variable has been initialized.

##### Args:


*  <b>`variable`</b>: A `Variable`.

##### Returns:

  An operation to check whether a variable has been initialized.


- - -

### `tf.assert_variables_initialized(var_list=None)` {#assert_variables_initialized}

Returns an Op to check if variables are initialized.

When run, the returned Op will raise the exception `FailedPreconditionError`
if any of the variables has not yet been initialized.

Note: This function is implemented by trying to fetch the values of the
variables. If one of the variables is not initialized a message may be
logged by the C++ runtime. This is expected.

##### Args:


*  <b>`var_list`</b>: List of `Variable` objects to check. Defaults to the
    value of `all_variables().`

##### Returns:

  An Op, or None if there are no variables.



## Saving and Restoring Variables

- - -

### `class tf.train.Saver` {#Saver}

Saves and restores variables.

See [Variables](../../how_tos/variables/index.md)
for an overview of variables, saving and restoring.

The `Saver` class adds ops to save and restore variables to and from
*checkpoints*.  It also provides convenience methods to run these ops.

Checkpoints are binary files in a proprietary format which map variable names
to tensor values.  The best way to examine the contents of a checkpoint is to
load it using a `Saver`.

Savers can automatically number checkpoint filenames with a provided counter.
This lets you keep multiple checkpoints at different steps while training a
model.  For example you can number the checkpoint filenames with the training
step number.  To avoid filling up disks, savers manage checkpoint files
automatically. For example, they can keep only the N most recent files, or
one checkpoint for every N hours of training.

You number checkpoint filenames by passing a value to the optional
`global_step` argument to `save()`:

```python
saver.save(sess, 'my-model', global_step=0) ==> filename: 'my-model-0'
...
saver.save(sess, 'my-model', global_step=1000) ==> filename: 'my-model-1000'
```

Additionally, optional arguments to the `Saver()` constructor let you control
the proliferation of checkpoint files on disk:

* `max_to_keep` indicates the maximum number of recent checkpoint files to
  keep.  As new files are created, older files are deleted.  If None or 0,
  all checkpoint files are kept.  Defaults to 5 (that is, the 5 most recent
  checkpoint files are kept.)

* `keep_checkpoint_every_n_hours`: In addition to keeping the most recent
  `max_to_keep` checkpoint files, you might want to keep one checkpoint file
  for every N hours of training.  This can be useful if you want to later
  analyze how a model progressed during a long training session.  For
  example, passing `keep_checkpoint_every_n_hours=2` ensures that you keep
  one checkpoint file for every 2 hours of training.  The default value of
  10,000 hours effectively disables the feature.

Note that you still have to call the `save()` method to save the model.
Passing these arguments to the constructor will not save variables
automatically for you.

A training program that saves regularly looks like:

```python
...
# Create a saver.
saver = tf.train.Saver(...variables...)
# Launch the graph and train, saving the model every 1,000 steps.
sess = tf.Session()
for step in xrange(1000000):
    sess.run(..training_op..)
    if step % 1000 == 0:
        # Append the step number to the checkpoint name:
        saver.save(sess, 'my-model', global_step=step)
```

In addition to checkpoint files, savers keep a protocol buffer on disk with
the list of recent checkpoints. This is used to manage numbered checkpoint
files and by `latest_checkpoint()`, which makes it easy to discover the path
to the most recent checkpoint. That protocol buffer is stored in a file named
'checkpoint' next to the checkpoint files.

If you create several savers, you can specify a different filename for the
protocol buffer file in the call to `save()`.

- - -

#### `tf.train.Saver.__init__(var_list=None, reshape=False, sharded=False, max_to_keep=5, keep_checkpoint_every_n_hours=10000.0, name=None, restore_sequentially=False, saver_def=None, builder=None)` {#Saver.__init__}

Creates a `Saver`.

The constructor adds ops to save and restore variables.

`var_list` specifies the variables that will be saved and restored. It can
be passed as a `dict` or a list:

* A `dict` of names to variables: The keys are the names that will be
  used to save or restore the variables in the checkpoint files.
* A list of variables: The variables will be keyed with their op name in
  the checkpoint files.

For example:

```python
v1 = tf.Variable(..., name='v1')
v2 = tf.Variable(..., name='v2')

# Pass the variables as a dict:
saver = tf.train.Saver({'v1': v1, 'v2': v2})

# Or pass them as a list.
saver = tf.train.Saver([v1, v2])
# Passing a list is equivalent to passing a dict with the variable op names
# as keys:
saver = tf.train.Saver({v.op.name: v for v in [v1, v2]})
```

The optional `reshape` argument, if `True`, allows restoring a variable from
a save file where the variable had a different shape, but the same number
of elements and type.  This is useful if you have reshaped a variable and
want to reload it from an older checkpoint.

The optional `sharded` argument, if `True`, instructs the saver to shard
checkpoints per device.

##### Args:


*  <b>`var_list`</b>: A list of `Variable` objects or a dictionary mapping names to
    variables.  If `None`, defaults to the list of all variables.
*  <b>`reshape`</b>: If `True`, allows restoring parameters from a checkpoint
    where the variables have a different shape.
*  <b>`sharded`</b>: If `True`, shard the checkpoints, one per device.
*  <b>`max_to_keep`</b>: Maximum number of recent checkpoints to keep.
    Defaults to 5.
*  <b>`keep_checkpoint_every_n_hours`</b>: How often to keep checkpoints.
    Defaults to 10,000 hours.
*  <b>`name`</b>: String.  Optional name to use as a prefix when adding operations.
*  <b>`restore_sequentially`</b>: A `Bool`, which if true, causes restore of different
    variables to happen sequentially within each device.  This can lower
    memory usage when restoring very large models.
*  <b>`saver_def`</b>: Optional `SaverDef` proto to use instead of running the
    builder. This is only useful for specialty code that wants to recreate
    a `Saver` object for a previously built `Graph` that had a `Saver`.
    The `saver_def` proto should be the one returned by the
    `as_saver_def()` call of the `Saver` that was created for that `Graph`.
*  <b>`builder`</b>: Optional `SaverBuilder` to use if a `saver_def` was not provided.
    Defaults to `BaseSaverBuilder()`.

##### Raises:


*  <b>`TypeError`</b>: If `var_list` is invalid.
*  <b>`ValueError`</b>: If any of the keys or values in `var_list` are not unique.


- - -

#### `tf.train.Saver.save(sess, save_path, global_step=None, latest_filename=None, meta_graph_suffix='meta', write_meta_graph=True)` {#Saver.save}

Saves variables.

This method runs the ops added by the constructor for saving variables.
It requires a session in which the graph was launched.  The variables to
save must also have been initialized.

The method returns the path of the newly created checkpoint file.  This
path can be passed directly to a call to `restore()`.

##### Args:


*  <b>`sess`</b>: A Session to use to save the variables.
*  <b>`save_path`</b>: String.  Path to the checkpoint filename.  If the saver is
    `sharded`, this is the prefix of the sharded checkpoint filename.
*  <b>`global_step`</b>: If provided the global step number is appended to
    `save_path` to create the checkpoint filename. The optional argument
    can be a `Tensor`, a `Tensor` name or an integer.
*  <b>`latest_filename`</b>: Optional name for the protocol buffer file that will
    contains the list of most recent checkpoint filenames.  That file,
    kept in the same directory as the checkpoint files, is automatically
    managed by the saver to keep track of recent checkpoints.  Defaults to
    'checkpoint'.
*  <b>`meta_graph_suffix`</b>: Suffix for `MetaGraphDef` file. Defaults to 'meta'.
*  <b>`write_meta_graph`</b>: `Boolean` indicating whether or not to write the meta
    graph file.

##### Returns:

  A string: path at which the variables were saved.  If the saver is
    sharded, this string ends with: '-?????-of-nnnnn' where 'nnnnn'
    is the number of shards created.

##### Raises:


*  <b>`TypeError`</b>: If `sess` is not a `Session`.
*  <b>`ValueError`</b>: If `latest_filename` contains path components.


- - -

#### `tf.train.Saver.restore(sess, save_path)` {#Saver.restore}

Restores previously saved variables.

This method runs the ops added by the constructor for restoring variables.
It requires a session in which the graph was launched.  The variables to
restore do not have to have been initialized, as restoring is itself a way
to initialize variables.

The `save_path` argument is typically a value previously returned from a
`save()` call, or a call to `latest_checkpoint()`.

##### Args:


*  <b>`sess`</b>: A `Session` to use to restore the parameters.
*  <b>`save_path`</b>: Path where parameters were previously saved.

##### Raises:


*  <b>`ValueError`</b>: If the given `save_path` does not point to a file.



Other utility methods.

- - -

#### `tf.train.Saver.last_checkpoints` {#Saver.last_checkpoints}

List of not-yet-deleted checkpoint filenames.

You can pass any of the returned values to `restore()`.

##### Returns:

  A list of checkpoint filenames, sorted from oldest to newest.


- - -

#### `tf.train.Saver.set_last_checkpoints(last_checkpoints)` {#Saver.set_last_checkpoints}

DEPRECATED: Use set_last_checkpoints_with_time.

Sets the list of old checkpoint filenames.

##### Args:


*  <b>`last_checkpoints`</b>: A list of checkpoint filenames.

##### Raises:


*  <b>`AssertionError`</b>: If last_checkpoints is not a list.


- - -

#### `tf.train.Saver.as_saver_def()` {#Saver.as_saver_def}

Generates a `SaverDef` representation of this saver.

##### Returns:

  A `SaverDef` proto.



#### Other Methods
- - -

#### `tf.train.Saver.export_meta_graph(filename=None, collection_list=None, as_text=False)` {#Saver.export_meta_graph}

Writes `MetaGraphDef` to save_path/filename.

##### Args:


*  <b>`filename`</b>: Optional meta_graph filename including the path.
*  <b>`collection_list`</b>: List of string keys to collect.
*  <b>`as_text`</b>: If `True`, writes the meta_graph as an ASCII proto.

##### Returns:

  A `MetaGraphDef` proto.


- - -

#### `tf.train.Saver.from_proto(saver_def)` {#Saver.from_proto}

Returns a `Saver` object created from `saver_def`.


- - -

#### `tf.train.Saver.set_last_checkpoints_with_time(last_checkpoints_with_time)` {#Saver.set_last_checkpoints_with_time}

Sets the list of old checkpoint filenames and timestamps.

##### Args:


*  <b>`last_checkpoints_with_time`</b>: A list of tuples of checkpoint filenames and
    timestamps.

##### Raises:


*  <b>`AssertionError`</b>: If last_checkpoints_with_time is not a list.


- - -

#### `tf.train.Saver.to_proto()` {#Saver.to_proto}

Converts this `Saver` to a `SaverDef` protocol buffer.

##### Returns:

  A `SaverDef` protocol buffer.




- - -

### `tf.train.latest_checkpoint(checkpoint_dir, latest_filename=None)` {#latest_checkpoint}

Finds the filename of latest saved checkpoint file.

##### Args:


*  <b>`checkpoint_dir`</b>: Directory where the variables were saved.
*  <b>`latest_filename`</b>: Optional name for the protocol buffer file that
    contains the list of most recent checkpoint filenames.
    See the corresponding argument to `Saver.save()`.

##### Returns:

  The full path to the latest checkpoint or `None` if no checkpoint was found.



- - -

### `tf.train.get_checkpoint_state(checkpoint_dir, latest_filename=None)` {#get_checkpoint_state}

Returns CheckpointState proto from the "checkpoint" file.

If the "checkpoint" file contains a valid CheckpointState
proto, returns it.

##### Args:


*  <b>`checkpoint_dir`</b>: The directory of checkpoints.
*  <b>`latest_filename`</b>: Optional name of the checkpoint file.  Default to
    'checkpoint'.

##### Returns:

  A CheckpointState if the state was available, None
  otherwise.


- - -

### `tf.train.update_checkpoint_state(save_dir, model_checkpoint_path, all_model_checkpoint_paths=None, latest_filename=None)` {#update_checkpoint_state}

Updates the content of the 'checkpoint' file.

This updates the checkpoint file containing a CheckpointState
proto.

##### Args:


*  <b>`save_dir`</b>: Directory where the model was saved.
*  <b>`model_checkpoint_path`</b>: The checkpoint file.
*  <b>`all_model_checkpoint_paths`</b>: List of strings.  Paths to all not-yet-deleted
    checkpoints, sorted from oldest to newest.  If this is a non-empty list,
    the last element must be equal to model_checkpoint_path.  These paths
    are also saved in the CheckpointState proto.
*  <b>`latest_filename`</b>: Optional name of the checkpoint file.  Default to
    'checkpoint'.

##### Raises:


*  <b>`RuntimeError`</b>: If the save paths conflict.



## Sharing Variables

TensorFlow provides several classes and operations that you can use to
create variables contingent on certain conditions.

- - -

### `tf.get_variable(name, shape=None, dtype=tf.float32, initializer=None, regularizer=None, trainable=True, collections=None, caching_device=None, partitioner=None, validate_shape=True)` {#get_variable}

Gets an existing variable with these parameters or create a new one.

This function prefixes the name with the current variable scope
and performs reuse checks. See the
[Variable Scope How To](../../how_tos/variable_scope/index.md)
for an extensive description of how reusing works. Here is a basic example:

```python
with tf.variable_scope("foo"):
    v = tf.get_variable("v", [1])  # v.name == "foo/v:0"
    w = tf.get_variable("w", [1])  # w.name == "foo/w:0"
with tf.variable_scope("foo", reuse=True)
    v1 = tf.get_variable("v")  # The same as v above.
```

If initializer is `None` (the default), the default initializer passed in
the variable scope will be used. If that one is `None` too, a
`UniformUnitScalingInitializer` will be used. The initializer can also be
a Tensor, in which case the variable is initialized to this value and shape.

Similarly, if the regularizer is `None` (the default), the default regularizer
passed in the variable scope will be used (if that is `None` too,
then by default no regularization is performed).

If a partitioner is provided, first a sharded `Variable` is created
via `_get_partitioned_variable_list`, and the return value is a
`Tensor` composed of the shards concatenated along the partition axis.

Some useful partitioners are available.  See, e.g.,
`variable_axis_size_partitioner`.

##### Args:


*  <b>`name`</b>: The name of the new or existing variable.
*  <b>`shape`</b>: Shape of the new or existing variable.
*  <b>`dtype`</b>: Type of the new or existing variable (defaults to `DT_FLOAT`).
*  <b>`initializer`</b>: Initializer for the variable if one is created.
*  <b>`regularizer`</b>: A (Tensor -> Tensor or None) function; the result of
    applying it on a newly created variable will be added to the collection
    GraphKeys.REGULARIZATION_LOSSES and can be used for regularization.
*  <b>`trainable`</b>: If `True` also add the variable to the graph collection
    `GraphKeys.TRAINABLE_VARIABLES` (see tf.Variable).
*  <b>`collections`</b>: List of graph collections keys to add the Variable to.
    Defaults to `[GraphKeys.VARIABLES]` (see tf.Variable).
    If partitioning is enabled and used, the concatenated return value
    is also added to collection `GraphKeys.CONCATENATED_VARIABLES`.
*  <b>`caching_device`</b>: Optional device string or function describing where the
    Variable should be cached for reading.  Defaults to the Variable's
    device.  If not `None`, caches on another device.  Typical use is to
    cache on the device where the Ops using the Variable reside, to
    deduplicate copying through `Switch` and other conditional statements.
*  <b>`partitioner`</b>: Optional callable that accepts a fully defined `TensorShape`
    and `dtype` of the Variable to be created, and returns a list of
    partitions for each axis (currently only one axis can be partitioned).
*  <b>`validate_shape`</b>: If False, allows the variable to be initialized with a
      value of unknown shape. If True, the default, the shape of initial_value
      must be known.

##### Returns:

  The created or existing variable.

##### Raises:


*  <b>`ValueError`</b>: when creating a new variable and shape is not declared,
    or when violating reuse during variable creation. Reuse is set inside
    `variable_scope`.


- - -

### `class tf.VariableScope` {#VariableScope}

Variable scope object to carry defaults to provide to get_variable.

Many of the arguments we need for get_variable in a variable store are most
easily handled with a context. This object is used for the defaults.

Attributes:
  name: name of the current scope, used as prefix in get_variable.
  initializer: default initializer passed to get_variable.
  regularizer: default regularizer passed to get_variable.
  reuse: Boolean or None, setting the reuse in get_variable.
  caching_device: string, callable, or None: the caching device passed to
    get_variable.
  partitioner: callable or `None`: the partitioner passed to `get_variable`.
  name_scope: The name passed to `tf.name_scope`.
- - -

#### `tf.VariableScope.__init__(reuse, name='', initializer=None, regularizer=None, caching_device=None, partitioner=None, name_scope='')` {#VariableScope.__init__}

Creates a new VariableScope with the given properties.


- - -

#### `tf.VariableScope.caching_device` {#VariableScope.caching_device}




- - -

#### `tf.VariableScope.get_variable(var_store, name, shape=None, dtype=tf.float32, initializer=None, regularizer=None, trainable=True, collections=None, caching_device=None, partitioner=None, validate_shape=True)` {#VariableScope.get_variable}

Gets an existing variable with this name or create a new one.


- - -

#### `tf.VariableScope.initializer` {#VariableScope.initializer}




- - -

#### `tf.VariableScope.name` {#VariableScope.name}




- - -

#### `tf.VariableScope.partitioner` {#VariableScope.partitioner}




- - -

#### `tf.VariableScope.regularizer` {#VariableScope.regularizer}




- - -

#### `tf.VariableScope.reuse` {#VariableScope.reuse}




- - -

#### `tf.VariableScope.reuse_variables()` {#VariableScope.reuse_variables}

Reuse variables in this scope.


- - -

#### `tf.VariableScope.set_caching_device(caching_device)` {#VariableScope.set_caching_device}

Set caching_device for this scope.


- - -

#### `tf.VariableScope.set_initializer(initializer)` {#VariableScope.set_initializer}

Set initializer for this scope.


- - -

#### `tf.VariableScope.set_partitioner(partitioner)` {#VariableScope.set_partitioner}

Set partitioner for this scope.


- - -

#### `tf.VariableScope.set_regularizer(regularizer)` {#VariableScope.set_regularizer}

Set regularizer for this scope.



- - -

### `tf.variable_scope(name_or_scope, reuse=None, initializer=None, regularizer=None, caching_device=None, partitioner=None)` {#variable_scope}

Returns a context for variable scope.

Variable scope allows to create new variables and to share already created
ones while providing checks to not create or share by accident. For details,
see the [Variable Scope How To](../../how_tos/variable_scope/index.md),
here we present only a few basic examples.

Simple example of how to create a new variable:

```python
with tf.variable_scope("foo"):
    with tf.variable_scope("bar"):
        v = tf.get_variable("v", [1])
        assert v.name == "foo/bar/v:0"
```

Basic example of sharing a variable:

```python
with tf.variable_scope("foo"):
    v = tf.get_variable("v", [1])
with tf.variable_scope("foo", reuse=True):
    v1 = tf.get_variable("v", [1])
assert v1 == v
```

Sharing a variable by capturing a scope and setting reuse:

```python
with tf.variable_scope("foo") as scope:
    v = tf.get_variable("v", [1])
    scope.reuse_variables()
    v1 = tf.get_variable("v", [1])
assert v1 == v
```

To prevent accidental sharing of variables, we raise an exception when
getting an existing variable in a non-reusing scope.

```python
with tf.variable_scope("foo"):
    v = tf.get_variable("v", [1])
    v1 = tf.get_variable("v", [1])
    #  Raises ValueError("... v already exists ...").
```

Similarly, we raise an exception when trying to get a variable that
does not exist in reuse mode.

```python
with tf.variable_scope("foo", reuse=True):
    v = tf.get_variable("v", [1])
    #  Raises ValueError("... v does not exists ...").
```

Note that the `reuse` flag is inherited: if we open a reusing scope,
then all its sub-scopes become reusing as well.

##### Args:


*  <b>`name_or_scope`</b>: `string` or `VariableScope`: the scope to open.
*  <b>`reuse`</b>: `True` or `None`; if `True`, we go into reuse mode for this scope as
    well as all sub-scopes; if `None`, we just inherit the parent scope reuse.
*  <b>`initializer`</b>: default initializer for variables within this scope.
*  <b>`regularizer`</b>: default regularizer for variables within this scope.
*  <b>`caching_device`</b>: default caching device for variables within this scope.
*  <b>`partitioner`</b>: default partitioner for variables within this scope.

##### Returns:

  A scope that can be to captured and reused.

##### Raises:


*  <b>`ValueError`</b>: when trying to reuse within a create scope, or create within
    a reuse scope, or if reuse is not `None` or `True`.
*  <b>`TypeError`</b>: when the types of some arguments are not appropriate.


- - -

### `tf.variable_op_scope(values, name_or_scope, default_name=None, initializer=None, regularizer=None, caching_device=None, partitioner=None, reuse=None)` {#variable_op_scope}

Returns a context manager for defining an op that creates variables.

This context manager validates that the given `values` are from the
same graph, ensures that graph is the default graph, and pushes a
name scope and a variable scope.

If `name_or_scope` is not None, it is used as is in the variable scope. If
`scope` is None, then `default_name` is used.  In that case, if the same name
has been previously used in the same scope, it will made unique be appending
`_N` to it.

This is intended to be used when defining generic ops and so reuse is always
inherited.

For example, to define a new Python op called `my_op_with_vars`:

```python
def my_op_with_vars(a, b, scope=None):
  with tf.variable_op_scope([a, b], scope, "MyOp") as scope:
    a = tf.convert_to_tensor(a, name="a")
    b = tf.convert_to_tensor(b, name="b")
    c = tf.get_variable('c')
    # Define some computation that uses `a`, `b`, and `c`.
    return foo_op(..., name=scope)
```

##### Args:


*  <b>`values`</b>: The list of `Tensor` arguments that are passed to the op function.
*  <b>`name_or_scope`</b>: The name argument that is passed to the op function,
    this name_or_scope is not uniquified in the variable scope.
*  <b>`default_name`</b>: The default name to use if the `name_or_scope` argument is
    `None`, this name will be uniquified. If name_or_scope is provided it
    won't be used and therefore it is not required and can be None.
*  <b>`initializer`</b>: The default initializer to pass to variable scope.
*  <b>`regularizer`</b>: The default regularizer for variables within this scope.
*  <b>`caching_device`</b>: The default caching device for variables within this scope.
*  <b>`partitioner`</b>: The default partitioner for variables within this scope.
*  <b>`reuse`</b>: `True` or `None`; if `True`, we go into reuse mode for this scope as
    well as all sub-scopes; if `None`, we just inherit the parent scope reuse.


##### Returns:

  A context manager for use in defining a Python op.

##### Raises:


*  <b>`ValueError`</b>: when trying to reuse within a create scope, or create within
    a reuse scope, or if reuse is not `None` or `True`.
*  <b>`TypeError`</b>: when the types of some arguments are not appropriate.


- - -

### `tf.get_variable_scope()` {#get_variable_scope}

Returns the current variable scope.


- - -

### `tf.make_template(name_, func_, **kwargs)` {#make_template}

Given an arbitrary function, wrap it so that it does variable sharing.

This wraps `func_` in a Template and partially evaluates it. Templates are
functions that create variables the first time they are called and reuse them
thereafter. In order for `func_` to be compatible with a `Template` it must
have the following properties:

* The function should create all trainable variables and any variables that
   should be reused by calling `tf.get_variable`. If a trainable variable is
   created using `tf.Variable`, then a ValueError will be thrown. Variables
   that are intended to be locals can be created by specifying
   `tf.Variable(..., trainable=false)`.
* The function may use variable scopes and other templates internally to
    create and reuse variables, but it shouldn't use `tf.get_variables` to
    capture variables that are defined outside of the scope of the function.
* Internal scopes and variable names should not depend on any arguments that
    are not supplied to `make_template`. In general you will get a ValueError
    telling you that you are trying to reuse a variable that doesn't exist
    if you make a mistake.

In the following example, both `z` and `w` will be scaled by the same `y`. It
is important to note that if we didn't assign `scalar_name` and used a
different name for z and w that a `ValueError` would be thrown because it
couldn't reuse the variable.

```python
def my_op(x, scalar_name):
  var1 = tf.get_variable(scalar_name,
                         shape=[],
                         initializer=tf.constant_initializer(1))
  return x * var1

scale_by_y = tf.make_template('scale_by_y', my_op, scalar_name='y')

z = scale_by_y(input1)
w = scale_by_y(input2)
```

As a safe-guard, the returned function will raise a `ValueError` after the
first call if trainable variables are created by calling `tf.Variable`.

If all of these are true, then 2 properties are enforced by the template:

1. Calling the same template multiple times will share all non-local
    variables.
2. Two different templates are guaranteed to be unique, unless you reenter the
    same variable scope as the initial definition of a template and redefine
    it. An examples of this exception:

```python
def my_op(x, scalar_name):
  var1 = tf.get_variable(scalar_name,
                         shape=[],
                         initializer=tf.constant_initializer(1))
  return x * var1

with tf.variable_scope('scope') as vs:
  scale_by_y = tf.make_template('scale_by_y', my_op, scalar_name='y')
  z = scale_by_y(input1)
  w = scale_by_y(input2)

# Creates a template that reuses the variables above.
with tf.variable_scope(vs, reuse=True):
  scale_by_y2 = tf.make_template('scale_by_y', my_op, scalar_name='y')
  z2 = scale_by_y2(input1)
  w2 = scale_by_y2(input2)
```

Note: The full variable scope is captured at the time of the first call.

Note: `name_` and `func_` have a following underscore to reduce the likelihood
of collisions with kwargs.

##### Args:


*  <b>`name_`</b>: A name for the scope created by this template. If necessary, the name
    will be made unique by appending `_N` to the name.
*  <b>`func_`</b>: The function to wrap.
*  <b>`**kwargs`</b>: Keyword arguments to apply to `func_`.

##### Returns:

  A function that will enter a `variable_scope` before calling `func_`. The
  first time it is called, it will create a non-reusing scope so that the
  variables will be unique.  On each subsequent call, it will reuse those
  variables.

##### Raises:


*  <b>`ValueError`</b>: if the name is None.



- - -

### `tf.no_regularizer(_)` {#no_regularizer}

Use this function to prevent regularization of variables.



- - -

### `tf.constant_initializer(value=0.0, dtype=tf.float32)` {#constant_initializer}

Returns an initializer that generates tensors with a single value.

##### Args:


*  <b>`value`</b>: A Python scalar. All elements of the initialized variable
    will be set to this value.
*  <b>`dtype`</b>: The data type. Only floating point types are supported.

##### Returns:

  An initializer that generates tensors with a single value.

##### Raises:


*  <b>`ValueError`</b>: if `dtype` is not a floating point type.


- - -

### `tf.random_normal_initializer(mean=0.0, stddev=1.0, seed=None, dtype=tf.float32)` {#random_normal_initializer}

Returns an initializer that generates tensors with a normal distribution.

##### Args:


*  <b>`mean`</b>: a python scalar or a scalar tensor. Mean of the random values
    to generate.
*  <b>`stddev`</b>: a python scalar or a scalar tensor. Standard deviation of the
    random values to generate.
*  <b>`seed`</b>: A Python integer. Used to create random seeds. See
    [`set_random_seed`](../../api_docs/python/constant_op.md#set_random_seed)
    for behavior.
*  <b>`dtype`</b>: The data type. Only floating point types are supported.

##### Returns:

  An initializer that generates tensors with a normal distribution.

##### Raises:


*  <b>`ValueError`</b>: if `dtype` is not a floating point type.


- - -

### `tf.truncated_normal_initializer(mean=0.0, stddev=1.0, seed=None, dtype=tf.float32)` {#truncated_normal_initializer}

Returns an initializer that generates a truncated normal distribution.

These values are similar to values from a `random_normal_initializer`
except that values more than two standard deviations from the mean
are discarded and re-drawn. This is the recommended initializer for
neural network weights and filters.

##### Args:


*  <b>`mean`</b>: a python scalar or a scalar tensor. Mean of the random values
    to generate.
*  <b>`stddev`</b>: a python scalar or a scalar tensor. Standard deviation of the
    random values to generate.
*  <b>`seed`</b>: A Python integer. Used to create random seeds. See
    [`set_random_seed`](../../api_docs/python/constant_op.md#set_random_seed)
    for behavior.
*  <b>`dtype`</b>: The data type. Only floating point types are supported.

##### Returns:

  An initializer that generates tensors with a truncated normal
  distribution.

##### Raises:


*  <b>`ValueError`</b>: if `dtype` is not a floating point type.


- - -

### `tf.random_uniform_initializer(minval=0.0, maxval=1.0, seed=None, dtype=tf.float32)` {#random_uniform_initializer}

Returns an initializer that generates tensors with a uniform distribution.

##### Args:


*  <b>`minval`</b>: a python scalar or a scalar tensor. lower bound of the range
    of random values to generate.
*  <b>`maxval`</b>: a python scalar or a scalar tensor. upper bound of the range
    of random values to generate.
*  <b>`seed`</b>: A Python integer. Used to create random seeds. See
    [`set_random_seed`](../../api_docs/python/constant_op.md#set_random_seed)
    for behavior.
*  <b>`dtype`</b>: The data type. Only floating point types are supported.

##### Returns:

  An initializer that generates tensors with a uniform distribution.

##### Raises:


*  <b>`ValueError`</b>: if `dtype` is not a floating point type.


- - -

### `tf.uniform_unit_scaling_initializer(factor=1.0, seed=None, dtype=tf.float32, full_shape=None)` {#uniform_unit_scaling_initializer}

Returns an initializer that generates tensors without scaling variance.

When initializing a deep network, it is in principle advantageous to keep
the scale of the input variance constant, so it does not explode or diminish
by reaching the final layer. If the input is `x` and the operation `x * W`,
and we want to initialize `W` uniformly at random, we need to pick `W` from

    [-sqrt(3) / sqrt(dim), sqrt(3) / sqrt(dim)]

to keep the scale intact, where `dim = W.shape[0]` (the size of the input).
A similar calculation for convolutional networks gives an analogous result
with `dim` equal to the product of the first 3 dimensions.  When
nonlinearities are present, we need to multiply this by a constant `factor`.
See [Sussillo et al., 2014](https://arxiv.org/abs/1412.6558)
([pdf](http://arxiv.org/pdf/1412.6558.pdf)) for deeper motivation, experiments
and the calculation of constants. In section 2.3 there, the constants were
numerically computed: for a linear layer it's 1.0, relu: ~1.43, tanh: ~1.15.

If the shape tuple `full_shape` is provided, the scale will be calculated from
this predefined shape.  This is useful when a `Variable` is being partitioned
across several shards, and each shard has a smaller shape than the whole.
Since the shards are usually concatenated when used, the scale should be
based on the shape of the whole.

##### Args:


*  <b>`factor`</b>: Float.  A multiplicative factor by which the values will be scaled.
*  <b>`seed`</b>: A Python integer. Used to create random seeds. See
    [`set_random_seed`](../../api_docs/python/constant_op.md#set_random_seed)
    for behavior.
*  <b>`dtype`</b>: The data type. Only floating point types are supported.
*  <b>`full_shape`</b>: Tuple or list of integers.  The shape used for calculating
    scale normalization (instead of the shape passed at creation time).
    Useful when creating sharded variables via partitioning.

##### Returns:

  An initializer that generates tensors with unit variance.

##### Raises:


*  <b>`ValueError`</b>: if `dtype` is not a floating point type.


- - -

### `tf.zeros_initializer(shape, dtype=tf.float32)` {#zeros_initializer}

An adaptor for zeros() to match the Initializer spec.


- - -

### `tf.ones_initializer(shape, dtype=tf.float32)` {#ones_initializer}

An adaptor for ones() to match the Initializer spec.



## Variable Partitioners for Sharding

- - -

### `tf.variable_axis_size_partitioner(max_shard_bytes, axis=0, bytes_per_string_element=16)` {#variable_axis_size_partitioner}

Get a partitioner for VariableScope to keep shards below `max_shard_bytes`.

This partitioner will shard a Variable along one axis, attempting to keep
the maximum shard size below `max_shard_bytes`.  In practice, this is not
always possible when sharding along only one axis.  When this happens,
this axis is sharded as much as possible (i.e., every dimension becomes
a separate shard).

One reasonable value for `max_shard_bytes` is `(64 << 20) - 1`, or almost
`64MB`, to keep below the protobuf byte limit.

##### Args:


*  <b>`max_shard_bytes`</b>: The maximum size any given shard is allowed to be.
*  <b>`axis`</b>: The axis to partition along.  Default: outermost axis.
*  <b>`bytes_per_string_element`</b>: If the `Variable` is of type string, this provides
    an estimate of how large each scalar in the `Variable` is.

##### Returns:

  A partition function usable as the `partitioner` argument to
  `variable_scope`, `get_variable`, and `get_partitioned_variable_list`.

##### Raises:


*  <b>`ValueError`</b>: If any of the byte counts are non-positive.



## Sparse Variable Updates

The sparse update ops modify a subset of the entries in a dense `Variable`,
either overwriting the entries or adding / subtracting a delta.  These are
useful for training embedding models and similar lookup-based networks, since
only a small subset of embedding vectors change in any given step.

Since a sparse update of a large tensor may be generated automatically during
gradient computation (as in the gradient of
[`tf.gather`](../../api_docs/python/array_ops.md#gather)),
an [`IndexedSlices`](#IndexedSlices) class is provided that encapsulates a set
of sparse indices and values.  `IndexedSlices` objects are detected and handled
automatically by the optimizers in most cases.

- - -

### `tf.scatter_update(ref, indices, updates, use_locking=None, name=None)` {#scatter_update}

Applies sparse updates to a variable reference.

This operation computes

    # Scalar indices
    ref[indices, ...] = updates[...]

    # Vector indices (for each i)
    ref[indices[i], ...] = updates[i, ...]

    # High rank indices (for each i, ..., j)
    ref[indices[i, ..., j], ...] = updates[i, ..., j, ...]

This operation outputs `ref` after the update is done.
This makes it easier to chain operations that need to use the reset value.

If values in `ref` is to be updated more than once, because there are
duplicate entires in `indices`, the order at which the updates happen
for each value is undefined.

Requires `updates.shape = indices.shape + ref.shape[1:]`.

<div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
<img style="width:100%" src="../../images/ScatterUpdate.png" alt>
</div>

##### Args:


*  <b>`ref`</b>: A mutable `Tensor`. Should be from a `Variable` node.
*  <b>`indices`</b>: A `Tensor`. Must be one of the following types: `int32`, `int64`.
    A tensor of indices into the first dimension of `ref`.
*  <b>`updates`</b>: A `Tensor`. Must have the same type as `ref`.
    A tensor of updated values to store in `ref`.
*  <b>`use_locking`</b>: An optional `bool`. Defaults to `True`.
    If True, the assignment will be protected by a lock;
    otherwise the behavior is undefined, but may exhibit less contention.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  Same as `ref`.  Returned as a convenience for operations that want
  to use the updated values after the update is done.


- - -

### `tf.scatter_add(ref, indices, updates, use_locking=None, name=None)` {#scatter_add}

Adds sparse updates to a variable reference.

This operation computes

    # Scalar indices
    ref[indices, ...] += updates[...]

    # Vector indices (for each i)
    ref[indices[i], ...] += updates[i, ...]

    # High rank indices (for each i, ..., j)
    ref[indices[i, ..., j], ...] += updates[i, ..., j, ...]

This operation outputs `ref` after the update is done.
This makes it easier to chain operations that need to use the reset value.

Duplicate entries are handled correctly: if multiple `indices` reference
the same location, their contributions add.

Requires `updates.shape = indices.shape + ref.shape[1:]`.

<div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
<img style="width:100%" src="../../images/ScatterAdd.png" alt>
</div>

##### Args:


*  <b>`ref`</b>: A mutable `Tensor`. Must be one of the following types: `float32`, `float64`, `int64`, `int32`, `uint8`, `uint16`, `int16`, `int8`, `complex64`, `complex128`, `qint8`, `quint8`, `qint32`, `half`.
    Should be from a `Variable` node.
*  <b>`indices`</b>: A `Tensor`. Must be one of the following types: `int32`, `int64`.
    A tensor of indices into the first dimension of `ref`.
*  <b>`updates`</b>: A `Tensor`. Must have the same type as `ref`.
    A tensor of updated values to add to `ref`.
*  <b>`use_locking`</b>: An optional `bool`. Defaults to `False`.
    If True, the addition will be protected by a lock;
    otherwise the behavior is undefined, but may exhibit less contention.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  Same as `ref`.  Returned as a convenience for operations that want
  to use the updated values after the update is done.


- - -

### `tf.scatter_sub(ref, indices, updates, use_locking=None, name=None)` {#scatter_sub}

Subtracts sparse updates to a variable reference.

    # Scalar indices
    ref[indices, ...] -= updates[...]

    # Vector indices (for each i)
    ref[indices[i], ...] -= updates[i, ...]

    # High rank indices (for each i, ..., j)
    ref[indices[i, ..., j], ...] -= updates[i, ..., j, ...]

This operation outputs `ref` after the update is done.
This makes it easier to chain operations that need to use the reset value.

Duplicate entries are handled correctly: if multiple `indices` reference
the same location, their (negated) contributions add.

Requires `updates.shape = indices.shape + ref.shape[1:]`.

<div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
<img style="width:100%" src="../../images/ScatterSub.png" alt>
</div>

##### Args:


*  <b>`ref`</b>: A mutable `Tensor`. Must be one of the following types: `float32`, `float64`, `int64`, `int32`, `uint8`, `uint16`, `int16`, `int8`, `complex64`, `complex128`, `qint8`, `quint8`, `qint32`, `half`.
    Should be from a `Variable` node.
*  <b>`indices`</b>: A `Tensor`. Must be one of the following types: `int32`, `int64`.
    A tensor of indices into the first dimension of `ref`.
*  <b>`updates`</b>: A `Tensor`. Must have the same type as `ref`.
    A tensor of updated values to subtract from `ref`.
*  <b>`use_locking`</b>: An optional `bool`. Defaults to `False`.
    If True, the subtraction will be protected by a lock;
    otherwise the behavior is undefined, but may exhibit less contention.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  Same as `ref`.  Returned as a convenience for operations that want
  to use the updated values after the update is done.


- - -

### `tf.sparse_mask(a, mask_indices, name=None)` {#sparse_mask}

Masks elements of `IndexedSlices`.

Given an `IndexedSlices` instance `a`, returns another `IndexedSlices` that
contains a subset of the slices of `a`. Only the slices at indices specified
in `mask_indices` are returned.

This is useful when you need to extract a subset of slices in an
`IndexedSlices` object.

For example:

```python
# `a` contains slices at indices [12, 26, 37, 45] from a large tensor
# with shape [1000, 10]
a.indices => [12, 26, 37, 45]
tf.shape(a.values) => [4, 10]

# `b` will be the subset of `a` slices at its second and third indices, so
# we want to mask of its first and last indices (which are at absolute
# indices 12, 45)
b = tf.sparse_mask(a, [12, 45])

b.indices => [26, 37]
tf.shape(b.values) => [2, 10]

```

##### Args:

  * `a`: An `IndexedSlices` instance.
  * `mask_indices`: Indices of elements to mask.
  * `name`: A name for the operation (optional).

##### Returns:

  The masked `IndexedSlices` instance.


- - -

### `class tf.IndexedSlices` {#IndexedSlices}

A sparse representation of a set of tensor slices at given indices.

This class is a simple wrapper for a pair of `Tensor` objects:

* `values`: A `Tensor` of any dtype with shape `[D0, D1, ..., Dn]`.
* `indices`: A 1-D integer `Tensor` with shape `[D0]`.

An `IndexedSlices` is typically used to represent a subset of a larger
tensor `dense` of shape `[LARGE0, D1, .. , DN]` where `LARGE0 >> D0`.
The values in `indices` are the indices in the first dimension of
the slices that have been extracted from the larger tensor.

The dense tensor `dense` represented by an `IndexedSlices` `slices` has

```python
dense[slices.indices[i], :, :, :, ...] = slices.values[i, :, :, :, ...]
```

The `IndexedSlices` class is used principally in the definition of
gradients for operations that have sparse gradients
(e.g. [`tf.gather`](../../api_docs/python/array_ops.md#gather)).

Contrast this representation with
[`SparseTensor`](../../api_docs/python/sparse_ops.md#SparseTensor),
which uses multi-dimensional indices and scalar values.

- - -

#### `tf.IndexedSlices.__init__(values, indices, dense_shape=None)` {#IndexedSlices.__init__}

Creates an `IndexedSlices`.



- - -

#### `tf.IndexedSlices.values` {#IndexedSlices.values}

A `Tensor` containing the values of the slices.


- - -

#### `tf.IndexedSlices.indices` {#IndexedSlices.indices}

A 1-D `Tensor` containing the indices of the slices.


- - -

#### `tf.IndexedSlices.dense_shape` {#IndexedSlices.dense_shape}

A 1-D `Tensor` containing the shape of the corresponding dense tensor.



- - -

#### `tf.IndexedSlices.name` {#IndexedSlices.name}

The name of this `IndexedSlices`.


- - -

#### `tf.IndexedSlices.dtype` {#IndexedSlices.dtype}

The `DType` of elements in this tensor.


- - -

#### `tf.IndexedSlices.device` {#IndexedSlices.device}

The name of the device on which `values` will be produced, or `None`.


- - -

#### `tf.IndexedSlices.op` {#IndexedSlices.op}

The `Operation` that produces `values` as an output.



#### Other Methods
- - -

#### `tf.IndexedSlices.graph` {#IndexedSlices.graph}

The `Graph` that contains the values, indices, and shape tensors.



