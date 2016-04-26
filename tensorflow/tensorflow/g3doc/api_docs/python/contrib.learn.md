<!-- This file is machine generated: DO NOT EDIT! -->

# Learn (contrib)
[TOC]



## Other Functions and Classes
- - -

### `class tf.contrib.learn.RunConfig` {#RunConfig}

This class specifies the specific configurations for the run.

Parameters:
    tf_master: TensorFlow master. Empty string is default for local.
    num_cores: Number of cores to be used. (default: 4)
    verbose: Controls the verbosity, possible values:
             0: the algorithm and debug information is muted.
             1: trainer prints the progress.
             2: log device placement is printed.
    gpu_memory_fraction: Fraction of GPU memory used by the process on
        each GPU uniformly on the same machine.
    tf_random_seed: Random seed for TensorFlow initializers.
        Setting this value, allows consistency between reruns.
    keep_checkpoint_max: The maximum number of recent checkpoint files to keep.
        As new files are created, older files are deleted.
        If None or 0, all checkpoint files are kept.
        Defaults to 5 (that is, the 5 most recent checkpoint files are kept.)
    keep_checkpoint_every_n_hours: Number of hours between each checkpoint
        to be saved. The default value of 10,000 hours effectively disables the feature.

Attributes:
    tf_master: Tensorflow master.
    tf_config: Tensorflow Session Config proto.
    tf_random_seed: Tensorflow random seed.
    keep_checkpoint_max: Maximum number of checkpoints to keep.
    keep_checkpoint_every_n_hours: Number of hours between each checkpoint.
- - -

#### `tf.contrib.learn.RunConfig.__init__(tf_master='', num_cores=4, verbose=1, gpu_memory_fraction=1, tf_random_seed=42, keep_checkpoint_max=5, keep_checkpoint_every_n_hours=10000)` {#RunConfig.__init__}





- - -

### `class tf.contrib.learn.TensorFlowClassifier` {#TensorFlowClassifier}

TensorFlow Linear Classifier model.
- - -

#### `tf.contrib.learn.TensorFlowClassifier.__init__(n_classes, batch_size=32, steps=200, optimizer='Adagrad', learning_rate=0.1, class_weight=None, clip_gradients=5.0, continue_training=False, config=None, verbose=1)` {#TensorFlowClassifier.__init__}




- - -

#### `tf.contrib.learn.TensorFlowClassifier.bias_` {#TensorFlowClassifier.bias_}

Returns weights of the linear classifier.


- - -

#### `tf.contrib.learn.TensorFlowClassifier.fit(X, y, monitor=None, logdir=None)` {#TensorFlowClassifier.fit}

Builds a neural network model given provided `model_fn` and training
data X and y.

Note: called first time constructs the graph and initializers
variables. Consecutives times it will continue training the same model.
This logic follows partial_fit() interface in scikit-learn.

To restart learning, create new estimator.

##### Args:


*  <b>`X`</b>: matrix or tensor of shape [n_samples, n_features...]. Can be
    iterator that returns arrays of features. The training input
    samples for fitting the model.
*  <b>`y`</b>: vector or matrix [n_samples] or [n_samples, n_outputs]. Can be
    iterator that returns array of targets. The training target values
    (class labels in classification, real numbers in regression).
*  <b>`monitor`</b>: Monitor object to print training progress and invoke early stopping
*  <b>`logdir`</b>: the directory to save the log file that can be used for
    optional visualization.

##### Returns:

    Returns self.


- - -

#### `tf.contrib.learn.TensorFlowClassifier.get_params(deep=True)` {#TensorFlowClassifier.get_params}

Get parameters for this estimator.

Parameters
----------
deep: boolean, optional
    If True, will return the parameters for this estimator and
    contained subobjects that are estimators.

Returns
-------
params : mapping of string to any
    Parameter names mapped to their values.


- - -

#### `tf.contrib.learn.TensorFlowClassifier.get_tensor(name)` {#TensorFlowClassifier.get_tensor}

Returns tensor by name.

##### Args:


*  <b>`name`</b>: string, name of the tensor.

##### Returns:

    Tensor.


- - -

#### `tf.contrib.learn.TensorFlowClassifier.get_tensor_value(name)` {#TensorFlowClassifier.get_tensor_value}

Returns value of the tensor give by name.

##### Args:


*  <b>`name`</b>: string, name of the tensor.

##### Returns:

    Numpy array - value of the tensor.


- - -

#### `tf.contrib.learn.TensorFlowClassifier.partial_fit(X, y)` {#TensorFlowClassifier.partial_fit}

Incremental fit on a batch of samples.

This method is expected to be called several times consecutively
on different or the same chunks of the dataset. This either can
implement iterative training or out-of-core/online training.

This is especially useful when the whole dataset is too big to
fit in memory at the same time. Or when model is taking long time
to converge, and you want to split up training into subparts.

##### Args:


*  <b>`X`</b>: matrix or tensor of shape [n_samples, n_features...]. Can be
    iterator that returns arrays of features. The training input
    samples for fitting the model.
*  <b>`y`</b>: vector or matrix [n_samples] or [n_samples, n_outputs]. Can be
    iterator that returns array of targets. The training target values
    (class label in classification, real numbers in regression).

##### Returns:

    Returns self.


- - -

#### `tf.contrib.learn.TensorFlowClassifier.predict(X, axis=1, batch_size=None)` {#TensorFlowClassifier.predict}

Predict class or regression for X.

For a classification model, the predicted class for each sample in X is
returned. For a regression model, the predicted value based on X is
returned.

##### Args:


*  <b>`X`</b>: array-like matrix, [n_samples, n_features...] or iterator.
*  <b>`axis`</b>: Which axis to argmax for classification.
          By default axis 1 (next after batch) is used.
          Use 2 for sequence predictions.
*  <b>`batch_size`</b>: If test set is too big, use batch size to split
                it into mini batches. By default the batch_size member
                variable is used.

##### Returns:


*  <b>`y`</b>: array of shape [n_samples]. The predicted classes or predicted
    value.


- - -

#### `tf.contrib.learn.TensorFlowClassifier.predict_proba(X, batch_size=None)` {#TensorFlowClassifier.predict_proba}

Predict class probability of the input samples X.

##### Args:


*  <b>`X`</b>: array-like matrix, [n_samples, n_features...] or iterator.
*  <b>`batch_size`</b>: If test set is too big, use batch size to split
                it into mini batches. By default the batch_size
                member variable is used.

##### Returns:


*  <b>`y`</b>: array of shape [n_samples, n_classes]. The predicted
    probabilities for each class.


- - -

#### `tf.contrib.learn.TensorFlowClassifier.restore(cls, path, config=None)` {#TensorFlowClassifier.restore}

Restores model from give path.

##### Args:


*  <b>`path`</b>: Path to the checkpoints and other model information.
*  <b>`config`</b>: RunConfig object that controls the configurations of the session,
        e.g. num_cores, gpu_memory_fraction, etc. This is allowed to be reconfigured.

##### Returns:

    Estiamator, object of the subclass of TensorFlowEstimator.


- - -

#### `tf.contrib.learn.TensorFlowClassifier.save(path)` {#TensorFlowClassifier.save}

Saves checkpoints and graph to given path.

##### Args:


*  <b>`path`</b>: Folder to save model to.


- - -

#### `tf.contrib.learn.TensorFlowClassifier.score(X, y, sample_weight=None)` {#TensorFlowClassifier.score}

Returns the mean accuracy on the given test data and labels.

In multi-label classification, this is the subset accuracy
which is a harsh metric since you require for each sample that
each label set be correctly predicted.

Parameters
----------
X : array-like, shape = (n_samples, n_features)
    Test samples.

y : array-like, shape = (n_samples) or (n_samples, n_outputs)
    True labels for X.

sample_weight : array-like, shape = [n_samples], optional
    Sample weights.

Returns
-------
score : float
    Mean accuracy of self.predict(X) wrt. y.


- - -

#### `tf.contrib.learn.TensorFlowClassifier.set_params(**params)` {#TensorFlowClassifier.set_params}

Set the parameters of this estimator.

The method works on simple estimators as well as on nested objects
(such as pipelines). The former have parameters of the form
``<component>__<parameter>`` so that it's possible to update each
component of a nested object.

Returns
-------
self


- - -

#### `tf.contrib.learn.TensorFlowClassifier.weights_` {#TensorFlowClassifier.weights_}

Returns weights of the linear classifier.



- - -

### `class tf.contrib.learn.TensorFlowDNNClassifier` {#TensorFlowDNNClassifier}

TensorFlow DNN Classifier model.

Parameters:
    hidden_units: List of hidden units per layer.
    n_classes: Number of classes in the target.
    batch_size: Mini batch size.
    steps: Number of steps to run over data.
    optimizer: Optimizer name (or class), for example "SGD", "Adam",
               "Adagrad".
    learning_rate: If this is constant float value, no decay function is used.
        Instead, a customized decay function can be passed that accepts
        global_step as parameter and returns a Tensor.
        e.g. exponential decay function:
        def exp_decay(global_step):
            return tf.train.exponential_decay(
                learning_rate=0.1, global_step,
                decay_steps=2, decay_rate=0.001)
    class_weight: None or list of n_classes floats. Weight associated with
                 classes for loss computation. If not given, all classes are suppose to have
                 weight one.
    continue_training: when continue_training is True, once initialized
        model will be continuely trained on every call of fit.
    config: RunConfig object that controls the configurations of the session,
        e.g. num_cores, gpu_memory_fraction, etc.
    dropout: When not None, the probability we will drop out a given
             coordinate.
- - -

#### `tf.contrib.learn.TensorFlowDNNClassifier.__init__(hidden_units, n_classes, batch_size=32, steps=200, optimizer='Adagrad', learning_rate=0.1, class_weight=None, clip_gradients=5.0, continue_training=False, config=None, verbose=1, dropout=None)` {#TensorFlowDNNClassifier.__init__}




- - -

#### `tf.contrib.learn.TensorFlowDNNClassifier.bias_` {#TensorFlowDNNClassifier.bias_}

Returns bias of the DNN's bias layers.


- - -

#### `tf.contrib.learn.TensorFlowDNNClassifier.fit(X, y, monitor=None, logdir=None)` {#TensorFlowDNNClassifier.fit}

Builds a neural network model given provided `model_fn` and training
data X and y.

Note: called first time constructs the graph and initializers
variables. Consecutives times it will continue training the same model.
This logic follows partial_fit() interface in scikit-learn.

To restart learning, create new estimator.

##### Args:


*  <b>`X`</b>: matrix or tensor of shape [n_samples, n_features...]. Can be
    iterator that returns arrays of features. The training input
    samples for fitting the model.
*  <b>`y`</b>: vector or matrix [n_samples] or [n_samples, n_outputs]. Can be
    iterator that returns array of targets. The training target values
    (class labels in classification, real numbers in regression).
*  <b>`monitor`</b>: Monitor object to print training progress and invoke early stopping
*  <b>`logdir`</b>: the directory to save the log file that can be used for
    optional visualization.

##### Returns:

    Returns self.


- - -

#### `tf.contrib.learn.TensorFlowDNNClassifier.get_params(deep=True)` {#TensorFlowDNNClassifier.get_params}

Get parameters for this estimator.

Parameters
----------
deep: boolean, optional
    If True, will return the parameters for this estimator and
    contained subobjects that are estimators.

Returns
-------
params : mapping of string to any
    Parameter names mapped to their values.


- - -

#### `tf.contrib.learn.TensorFlowDNNClassifier.get_tensor(name)` {#TensorFlowDNNClassifier.get_tensor}

Returns tensor by name.

##### Args:


*  <b>`name`</b>: string, name of the tensor.

##### Returns:

    Tensor.


- - -

#### `tf.contrib.learn.TensorFlowDNNClassifier.get_tensor_value(name)` {#TensorFlowDNNClassifier.get_tensor_value}

Returns value of the tensor give by name.

##### Args:


*  <b>`name`</b>: string, name of the tensor.

##### Returns:

    Numpy array - value of the tensor.


- - -

#### `tf.contrib.learn.TensorFlowDNNClassifier.partial_fit(X, y)` {#TensorFlowDNNClassifier.partial_fit}

Incremental fit on a batch of samples.

This method is expected to be called several times consecutively
on different or the same chunks of the dataset. This either can
implement iterative training or out-of-core/online training.

This is especially useful when the whole dataset is too big to
fit in memory at the same time. Or when model is taking long time
to converge, and you want to split up training into subparts.

##### Args:


*  <b>`X`</b>: matrix or tensor of shape [n_samples, n_features...]. Can be
    iterator that returns arrays of features. The training input
    samples for fitting the model.
*  <b>`y`</b>: vector or matrix [n_samples] or [n_samples, n_outputs]. Can be
    iterator that returns array of targets. The training target values
    (class label in classification, real numbers in regression).

##### Returns:

    Returns self.


- - -

#### `tf.contrib.learn.TensorFlowDNNClassifier.predict(X, axis=1, batch_size=None)` {#TensorFlowDNNClassifier.predict}

Predict class or regression for X.

For a classification model, the predicted class for each sample in X is
returned. For a regression model, the predicted value based on X is
returned.

##### Args:


*  <b>`X`</b>: array-like matrix, [n_samples, n_features...] or iterator.
*  <b>`axis`</b>: Which axis to argmax for classification.
          By default axis 1 (next after batch) is used.
          Use 2 for sequence predictions.
*  <b>`batch_size`</b>: If test set is too big, use batch size to split
                it into mini batches. By default the batch_size member
                variable is used.

##### Returns:


*  <b>`y`</b>: array of shape [n_samples]. The predicted classes or predicted
    value.


- - -

#### `tf.contrib.learn.TensorFlowDNNClassifier.predict_proba(X, batch_size=None)` {#TensorFlowDNNClassifier.predict_proba}

Predict class probability of the input samples X.

##### Args:


*  <b>`X`</b>: array-like matrix, [n_samples, n_features...] or iterator.
*  <b>`batch_size`</b>: If test set is too big, use batch size to split
                it into mini batches. By default the batch_size
                member variable is used.

##### Returns:


*  <b>`y`</b>: array of shape [n_samples, n_classes]. The predicted
    probabilities for each class.


- - -

#### `tf.contrib.learn.TensorFlowDNNClassifier.restore(cls, path, config=None)` {#TensorFlowDNNClassifier.restore}

Restores model from give path.

##### Args:


*  <b>`path`</b>: Path to the checkpoints and other model information.
*  <b>`config`</b>: RunConfig object that controls the configurations of the session,
        e.g. num_cores, gpu_memory_fraction, etc. This is allowed to be reconfigured.

##### Returns:

    Estiamator, object of the subclass of TensorFlowEstimator.


- - -

#### `tf.contrib.learn.TensorFlowDNNClassifier.save(path)` {#TensorFlowDNNClassifier.save}

Saves checkpoints and graph to given path.

##### Args:


*  <b>`path`</b>: Folder to save model to.


- - -

#### `tf.contrib.learn.TensorFlowDNNClassifier.score(X, y, sample_weight=None)` {#TensorFlowDNNClassifier.score}

Returns the mean accuracy on the given test data and labels.

In multi-label classification, this is the subset accuracy
which is a harsh metric since you require for each sample that
each label set be correctly predicted.

Parameters
----------
X : array-like, shape = (n_samples, n_features)
    Test samples.

y : array-like, shape = (n_samples) or (n_samples, n_outputs)
    True labels for X.

sample_weight : array-like, shape = [n_samples], optional
    Sample weights.

Returns
-------
score : float
    Mean accuracy of self.predict(X) wrt. y.


- - -

#### `tf.contrib.learn.TensorFlowDNNClassifier.set_params(**params)` {#TensorFlowDNNClassifier.set_params}

Set the parameters of this estimator.

The method works on simple estimators as well as on nested objects
(such as pipelines). The former have parameters of the form
``<component>__<parameter>`` so that it's possible to update each
component of a nested object.

Returns
-------
self


- - -

#### `tf.contrib.learn.TensorFlowDNNClassifier.weights_` {#TensorFlowDNNClassifier.weights_}

Returns weights of the DNN weight layers.



- - -

### `class tf.contrib.learn.TensorFlowDNNRegressor` {#TensorFlowDNNRegressor}

TensorFlow DNN Regressor model.

Parameters:
    hidden_units: List of hidden units per layer.
    batch_size: Mini batch size.
    steps: Number of steps to run over data.
    optimizer: Optimizer name (or class), for example "SGD", "Adam",
               "Adagrad".
    learning_rate: If this is constant float value, no decay function is used.
        Instead, a customized decay function can be passed that accepts
        global_step as parameter and returns a Tensor.
        e.g. exponential decay function:
        def exp_decay(global_step):
            return tf.train.exponential_decay(
                learning_rate=0.1, global_step,
                decay_steps=2, decay_rate=0.001)
    continue_training: when continue_training is True, once initialized
        model will be continuely trained on every call of fit.
    config: RunConfig object that controls the configurations of the session,
        e.g. num_cores, gpu_memory_fraction, etc.
    verbose: Controls the verbosity, possible values:
             0: the algorithm and debug information is muted.
             1: trainer prints the progress.
             2: log device placement is printed.
    dropout: When not None, the probability we will drop out a given
             coordinate.
- - -

#### `tf.contrib.learn.TensorFlowDNNRegressor.__init__(hidden_units, n_classes=0, batch_size=32, steps=200, optimizer='Adagrad', learning_rate=0.1, clip_gradients=5.0, continue_training=False, config=None, verbose=1, dropout=None)` {#TensorFlowDNNRegressor.__init__}




- - -

#### `tf.contrib.learn.TensorFlowDNNRegressor.bias_` {#TensorFlowDNNRegressor.bias_}

Returns bias of the DNN's bias layers.


- - -

#### `tf.contrib.learn.TensorFlowDNNRegressor.fit(X, y, monitor=None, logdir=None)` {#TensorFlowDNNRegressor.fit}

Builds a neural network model given provided `model_fn` and training
data X and y.

Note: called first time constructs the graph and initializers
variables. Consecutives times it will continue training the same model.
This logic follows partial_fit() interface in scikit-learn.

To restart learning, create new estimator.

##### Args:


*  <b>`X`</b>: matrix or tensor of shape [n_samples, n_features...]. Can be
    iterator that returns arrays of features. The training input
    samples for fitting the model.
*  <b>`y`</b>: vector or matrix [n_samples] or [n_samples, n_outputs]. Can be
    iterator that returns array of targets. The training target values
    (class labels in classification, real numbers in regression).
*  <b>`monitor`</b>: Monitor object to print training progress and invoke early stopping
*  <b>`logdir`</b>: the directory to save the log file that can be used for
    optional visualization.

##### Returns:

    Returns self.


- - -

#### `tf.contrib.learn.TensorFlowDNNRegressor.get_params(deep=True)` {#TensorFlowDNNRegressor.get_params}

Get parameters for this estimator.

Parameters
----------
deep: boolean, optional
    If True, will return the parameters for this estimator and
    contained subobjects that are estimators.

Returns
-------
params : mapping of string to any
    Parameter names mapped to their values.


- - -

#### `tf.contrib.learn.TensorFlowDNNRegressor.get_tensor(name)` {#TensorFlowDNNRegressor.get_tensor}

Returns tensor by name.

##### Args:


*  <b>`name`</b>: string, name of the tensor.

##### Returns:

    Tensor.


- - -

#### `tf.contrib.learn.TensorFlowDNNRegressor.get_tensor_value(name)` {#TensorFlowDNNRegressor.get_tensor_value}

Returns value of the tensor give by name.

##### Args:


*  <b>`name`</b>: string, name of the tensor.

##### Returns:

    Numpy array - value of the tensor.


- - -

#### `tf.contrib.learn.TensorFlowDNNRegressor.partial_fit(X, y)` {#TensorFlowDNNRegressor.partial_fit}

Incremental fit on a batch of samples.

This method is expected to be called several times consecutively
on different or the same chunks of the dataset. This either can
implement iterative training or out-of-core/online training.

This is especially useful when the whole dataset is too big to
fit in memory at the same time. Or when model is taking long time
to converge, and you want to split up training into subparts.

##### Args:


*  <b>`X`</b>: matrix or tensor of shape [n_samples, n_features...]. Can be
    iterator that returns arrays of features. The training input
    samples for fitting the model.
*  <b>`y`</b>: vector or matrix [n_samples] or [n_samples, n_outputs]. Can be
    iterator that returns array of targets. The training target values
    (class label in classification, real numbers in regression).

##### Returns:

    Returns self.


- - -

#### `tf.contrib.learn.TensorFlowDNNRegressor.predict(X, axis=1, batch_size=None)` {#TensorFlowDNNRegressor.predict}

Predict class or regression for X.

For a classification model, the predicted class for each sample in X is
returned. For a regression model, the predicted value based on X is
returned.

##### Args:


*  <b>`X`</b>: array-like matrix, [n_samples, n_features...] or iterator.
*  <b>`axis`</b>: Which axis to argmax for classification.
          By default axis 1 (next after batch) is used.
          Use 2 for sequence predictions.
*  <b>`batch_size`</b>: If test set is too big, use batch size to split
                it into mini batches. By default the batch_size member
                variable is used.

##### Returns:


*  <b>`y`</b>: array of shape [n_samples]. The predicted classes or predicted
    value.


- - -

#### `tf.contrib.learn.TensorFlowDNNRegressor.predict_proba(X, batch_size=None)` {#TensorFlowDNNRegressor.predict_proba}

Predict class probability of the input samples X.

##### Args:


*  <b>`X`</b>: array-like matrix, [n_samples, n_features...] or iterator.
*  <b>`batch_size`</b>: If test set is too big, use batch size to split
                it into mini batches. By default the batch_size
                member variable is used.

##### Returns:


*  <b>`y`</b>: array of shape [n_samples, n_classes]. The predicted
    probabilities for each class.


- - -

#### `tf.contrib.learn.TensorFlowDNNRegressor.restore(cls, path, config=None)` {#TensorFlowDNNRegressor.restore}

Restores model from give path.

##### Args:


*  <b>`path`</b>: Path to the checkpoints and other model information.
*  <b>`config`</b>: RunConfig object that controls the configurations of the session,
        e.g. num_cores, gpu_memory_fraction, etc. This is allowed to be reconfigured.

##### Returns:

    Estiamator, object of the subclass of TensorFlowEstimator.


- - -

#### `tf.contrib.learn.TensorFlowDNNRegressor.save(path)` {#TensorFlowDNNRegressor.save}

Saves checkpoints and graph to given path.

##### Args:


*  <b>`path`</b>: Folder to save model to.


- - -

#### `tf.contrib.learn.TensorFlowDNNRegressor.score(X, y, sample_weight=None)` {#TensorFlowDNNRegressor.score}

Returns the coefficient of determination R^2 of the prediction.

The coefficient R^2 is defined as (1 - u/v), where u is the regression
sum of squares ((y_true - y_pred) ** 2).sum() and v is the residual
sum of squares ((y_true - y_true.mean()) ** 2).sum().
Best possible score is 1.0 and it can be negative (because the
model can be arbitrarily worse). A constant model that always
predicts the expected value of y, disregarding the input features,
would get a R^2 score of 0.0.

Parameters
----------
X : array-like, shape = (n_samples, n_features)
    Test samples.

y : array-like, shape = (n_samples) or (n_samples, n_outputs)
    True values for X.

sample_weight : array-like, shape = [n_samples], optional
    Sample weights.

Returns
-------
score : float
    R^2 of self.predict(X) wrt. y.


- - -

#### `tf.contrib.learn.TensorFlowDNNRegressor.set_params(**params)` {#TensorFlowDNNRegressor.set_params}

Set the parameters of this estimator.

The method works on simple estimators as well as on nested objects
(such as pipelines). The former have parameters of the form
``<component>__<parameter>`` so that it's possible to update each
component of a nested object.

Returns
-------
self


- - -

#### `tf.contrib.learn.TensorFlowDNNRegressor.weights_` {#TensorFlowDNNRegressor.weights_}

Returns weights of the DNN weight layers.



- - -

### `class tf.contrib.learn.TensorFlowEstimator` {#TensorFlowEstimator}

Base class for all TensorFlow estimators.

Parameters:
    model_fn: Model function, that takes input X, y tensors and outputs
              prediction and loss tensors.
    n_classes: Number of classes in the target.
    batch_size: Mini batch size.
    steps: Number of steps to run over data.
    optimizer: Optimizer name (or class), for example "SGD", "Adam",
               "Adagrad".
    learning_rate: If this is constant float value, no decay function is used.
        Instead, a customized decay function can be passed that accepts
        global_step as parameter and returns a Tensor.
        e.g. exponential decay function:
        def exp_decay(global_step):
            return tf.train.exponential_decay(
                learning_rate=0.1, global_step,
                decay_steps=2, decay_rate=0.001)
    clip_gradients: Clip norm of the gradients to this value to stop
                    gradient explosion.
    class_weight: None or list of n_classes floats. Weight associated with
                 classes for loss computation. If not given, all classes are suppose to have
                 weight one.
    continue_training: when continue_training is True, once initialized
        model will be continuely trained on every call of fit.
    config: RunConfig object that controls the configurations of the session,
        e.g. num_cores, gpu_memory_fraction, etc.
    verbose: Controls the verbosity, possible values:
             0: the algorithm and debug information is muted.
             1: trainer prints the progress.
             2: log device placement is printed.
- - -

#### `tf.contrib.learn.TensorFlowEstimator.__init__(model_fn, n_classes, batch_size=32, steps=200, optimizer='Adagrad', learning_rate=0.1, clip_gradients=5.0, class_weight=None, continue_training=False, config=None, verbose=1)` {#TensorFlowEstimator.__init__}




- - -

#### `tf.contrib.learn.TensorFlowEstimator.fit(X, y, monitor=None, logdir=None)` {#TensorFlowEstimator.fit}

Builds a neural network model given provided `model_fn` and training
data X and y.

Note: called first time constructs the graph and initializers
variables. Consecutives times it will continue training the same model.
This logic follows partial_fit() interface in scikit-learn.

To restart learning, create new estimator.

##### Args:


*  <b>`X`</b>: matrix or tensor of shape [n_samples, n_features...]. Can be
    iterator that returns arrays of features. The training input
    samples for fitting the model.
*  <b>`y`</b>: vector or matrix [n_samples] or [n_samples, n_outputs]. Can be
    iterator that returns array of targets. The training target values
    (class labels in classification, real numbers in regression).
*  <b>`monitor`</b>: Monitor object to print training progress and invoke early stopping
*  <b>`logdir`</b>: the directory to save the log file that can be used for
    optional visualization.

##### Returns:

    Returns self.


- - -

#### `tf.contrib.learn.TensorFlowEstimator.get_params(deep=True)` {#TensorFlowEstimator.get_params}

Get parameters for this estimator.

Parameters
----------
deep: boolean, optional
    If True, will return the parameters for this estimator and
    contained subobjects that are estimators.

Returns
-------
params : mapping of string to any
    Parameter names mapped to their values.


- - -

#### `tf.contrib.learn.TensorFlowEstimator.get_tensor(name)` {#TensorFlowEstimator.get_tensor}

Returns tensor by name.

##### Args:


*  <b>`name`</b>: string, name of the tensor.

##### Returns:

    Tensor.


- - -

#### `tf.contrib.learn.TensorFlowEstimator.get_tensor_value(name)` {#TensorFlowEstimator.get_tensor_value}

Returns value of the tensor give by name.

##### Args:


*  <b>`name`</b>: string, name of the tensor.

##### Returns:

    Numpy array - value of the tensor.


- - -

#### `tf.contrib.learn.TensorFlowEstimator.partial_fit(X, y)` {#TensorFlowEstimator.partial_fit}

Incremental fit on a batch of samples.

This method is expected to be called several times consecutively
on different or the same chunks of the dataset. This either can
implement iterative training or out-of-core/online training.

This is especially useful when the whole dataset is too big to
fit in memory at the same time. Or when model is taking long time
to converge, and you want to split up training into subparts.

##### Args:


*  <b>`X`</b>: matrix or tensor of shape [n_samples, n_features...]. Can be
    iterator that returns arrays of features. The training input
    samples for fitting the model.
*  <b>`y`</b>: vector or matrix [n_samples] or [n_samples, n_outputs]. Can be
    iterator that returns array of targets. The training target values
    (class label in classification, real numbers in regression).

##### Returns:

    Returns self.


- - -

#### `tf.contrib.learn.TensorFlowEstimator.predict(X, axis=1, batch_size=None)` {#TensorFlowEstimator.predict}

Predict class or regression for X.

For a classification model, the predicted class for each sample in X is
returned. For a regression model, the predicted value based on X is
returned.

##### Args:


*  <b>`X`</b>: array-like matrix, [n_samples, n_features...] or iterator.
*  <b>`axis`</b>: Which axis to argmax for classification.
          By default axis 1 (next after batch) is used.
          Use 2 for sequence predictions.
*  <b>`batch_size`</b>: If test set is too big, use batch size to split
                it into mini batches. By default the batch_size member
                variable is used.

##### Returns:


*  <b>`y`</b>: array of shape [n_samples]. The predicted classes or predicted
    value.


- - -

#### `tf.contrib.learn.TensorFlowEstimator.predict_proba(X, batch_size=None)` {#TensorFlowEstimator.predict_proba}

Predict class probability of the input samples X.

##### Args:


*  <b>`X`</b>: array-like matrix, [n_samples, n_features...] or iterator.
*  <b>`batch_size`</b>: If test set is too big, use batch size to split
                it into mini batches. By default the batch_size
                member variable is used.

##### Returns:


*  <b>`y`</b>: array of shape [n_samples, n_classes]. The predicted
    probabilities for each class.


- - -

#### `tf.contrib.learn.TensorFlowEstimator.restore(cls, path, config=None)` {#TensorFlowEstimator.restore}

Restores model from give path.

##### Args:


*  <b>`path`</b>: Path to the checkpoints and other model information.
*  <b>`config`</b>: RunConfig object that controls the configurations of the session,
        e.g. num_cores, gpu_memory_fraction, etc. This is allowed to be reconfigured.

##### Returns:

    Estiamator, object of the subclass of TensorFlowEstimator.


- - -

#### `tf.contrib.learn.TensorFlowEstimator.save(path)` {#TensorFlowEstimator.save}

Saves checkpoints and graph to given path.

##### Args:


*  <b>`path`</b>: Folder to save model to.


- - -

#### `tf.contrib.learn.TensorFlowEstimator.set_params(**params)` {#TensorFlowEstimator.set_params}

Set the parameters of this estimator.

The method works on simple estimators as well as on nested objects
(such as pipelines). The former have parameters of the form
``<component>__<parameter>`` so that it's possible to update each
component of a nested object.

Returns
-------
self



- - -

### `class tf.contrib.learn.TensorFlowLinearClassifier` {#TensorFlowLinearClassifier}

TensorFlow Linear Classifier model.
- - -

#### `tf.contrib.learn.TensorFlowLinearClassifier.__init__(n_classes, batch_size=32, steps=200, optimizer='Adagrad', learning_rate=0.1, class_weight=None, clip_gradients=5.0, continue_training=False, config=None, verbose=1)` {#TensorFlowLinearClassifier.__init__}




- - -

#### `tf.contrib.learn.TensorFlowLinearClassifier.bias_` {#TensorFlowLinearClassifier.bias_}

Returns weights of the linear classifier.


- - -

#### `tf.contrib.learn.TensorFlowLinearClassifier.fit(X, y, monitor=None, logdir=None)` {#TensorFlowLinearClassifier.fit}

Builds a neural network model given provided `model_fn` and training
data X and y.

Note: called first time constructs the graph and initializers
variables. Consecutives times it will continue training the same model.
This logic follows partial_fit() interface in scikit-learn.

To restart learning, create new estimator.

##### Args:


*  <b>`X`</b>: matrix or tensor of shape [n_samples, n_features...]. Can be
    iterator that returns arrays of features. The training input
    samples for fitting the model.
*  <b>`y`</b>: vector or matrix [n_samples] or [n_samples, n_outputs]. Can be
    iterator that returns array of targets. The training target values
    (class labels in classification, real numbers in regression).
*  <b>`monitor`</b>: Monitor object to print training progress and invoke early stopping
*  <b>`logdir`</b>: the directory to save the log file that can be used for
    optional visualization.

##### Returns:

    Returns self.


- - -

#### `tf.contrib.learn.TensorFlowLinearClassifier.get_params(deep=True)` {#TensorFlowLinearClassifier.get_params}

Get parameters for this estimator.

Parameters
----------
deep: boolean, optional
    If True, will return the parameters for this estimator and
    contained subobjects that are estimators.

Returns
-------
params : mapping of string to any
    Parameter names mapped to their values.


- - -

#### `tf.contrib.learn.TensorFlowLinearClassifier.get_tensor(name)` {#TensorFlowLinearClassifier.get_tensor}

Returns tensor by name.

##### Args:


*  <b>`name`</b>: string, name of the tensor.

##### Returns:

    Tensor.


- - -

#### `tf.contrib.learn.TensorFlowLinearClassifier.get_tensor_value(name)` {#TensorFlowLinearClassifier.get_tensor_value}

Returns value of the tensor give by name.

##### Args:


*  <b>`name`</b>: string, name of the tensor.

##### Returns:

    Numpy array - value of the tensor.


- - -

#### `tf.contrib.learn.TensorFlowLinearClassifier.partial_fit(X, y)` {#TensorFlowLinearClassifier.partial_fit}

Incremental fit on a batch of samples.

This method is expected to be called several times consecutively
on different or the same chunks of the dataset. This either can
implement iterative training or out-of-core/online training.

This is especially useful when the whole dataset is too big to
fit in memory at the same time. Or when model is taking long time
to converge, and you want to split up training into subparts.

##### Args:


*  <b>`X`</b>: matrix or tensor of shape [n_samples, n_features...]. Can be
    iterator that returns arrays of features. The training input
    samples for fitting the model.
*  <b>`y`</b>: vector or matrix [n_samples] or [n_samples, n_outputs]. Can be
    iterator that returns array of targets. The training target values
    (class label in classification, real numbers in regression).

##### Returns:

    Returns self.


- - -

#### `tf.contrib.learn.TensorFlowLinearClassifier.predict(X, axis=1, batch_size=None)` {#TensorFlowLinearClassifier.predict}

Predict class or regression for X.

For a classification model, the predicted class for each sample in X is
returned. For a regression model, the predicted value based on X is
returned.

##### Args:


*  <b>`X`</b>: array-like matrix, [n_samples, n_features...] or iterator.
*  <b>`axis`</b>: Which axis to argmax for classification.
          By default axis 1 (next after batch) is used.
          Use 2 for sequence predictions.
*  <b>`batch_size`</b>: If test set is too big, use batch size to split
                it into mini batches. By default the batch_size member
                variable is used.

##### Returns:


*  <b>`y`</b>: array of shape [n_samples]. The predicted classes or predicted
    value.


- - -

#### `tf.contrib.learn.TensorFlowLinearClassifier.predict_proba(X, batch_size=None)` {#TensorFlowLinearClassifier.predict_proba}

Predict class probability of the input samples X.

##### Args:


*  <b>`X`</b>: array-like matrix, [n_samples, n_features...] or iterator.
*  <b>`batch_size`</b>: If test set is too big, use batch size to split
                it into mini batches. By default the batch_size
                member variable is used.

##### Returns:


*  <b>`y`</b>: array of shape [n_samples, n_classes]. The predicted
    probabilities for each class.


- - -

#### `tf.contrib.learn.TensorFlowLinearClassifier.restore(cls, path, config=None)` {#TensorFlowLinearClassifier.restore}

Restores model from give path.

##### Args:


*  <b>`path`</b>: Path to the checkpoints and other model information.
*  <b>`config`</b>: RunConfig object that controls the configurations of the session,
        e.g. num_cores, gpu_memory_fraction, etc. This is allowed to be reconfigured.

##### Returns:

    Estiamator, object of the subclass of TensorFlowEstimator.


- - -

#### `tf.contrib.learn.TensorFlowLinearClassifier.save(path)` {#TensorFlowLinearClassifier.save}

Saves checkpoints and graph to given path.

##### Args:


*  <b>`path`</b>: Folder to save model to.


- - -

#### `tf.contrib.learn.TensorFlowLinearClassifier.score(X, y, sample_weight=None)` {#TensorFlowLinearClassifier.score}

Returns the mean accuracy on the given test data and labels.

In multi-label classification, this is the subset accuracy
which is a harsh metric since you require for each sample that
each label set be correctly predicted.

Parameters
----------
X : array-like, shape = (n_samples, n_features)
    Test samples.

y : array-like, shape = (n_samples) or (n_samples, n_outputs)
    True labels for X.

sample_weight : array-like, shape = [n_samples], optional
    Sample weights.

Returns
-------
score : float
    Mean accuracy of self.predict(X) wrt. y.


- - -

#### `tf.contrib.learn.TensorFlowLinearClassifier.set_params(**params)` {#TensorFlowLinearClassifier.set_params}

Set the parameters of this estimator.

The method works on simple estimators as well as on nested objects
(such as pipelines). The former have parameters of the form
``<component>__<parameter>`` so that it's possible to update each
component of a nested object.

Returns
-------
self


- - -

#### `tf.contrib.learn.TensorFlowLinearClassifier.weights_` {#TensorFlowLinearClassifier.weights_}

Returns weights of the linear classifier.



- - -

### `class tf.contrib.learn.TensorFlowLinearRegressor` {#TensorFlowLinearRegressor}

TensorFlow Linear Regression model.
- - -

#### `tf.contrib.learn.TensorFlowLinearRegressor.__init__(n_classes=0, batch_size=32, steps=200, optimizer='Adagrad', learning_rate=0.1, clip_gradients=5.0, continue_training=False, config=None, verbose=1)` {#TensorFlowLinearRegressor.__init__}




- - -

#### `tf.contrib.learn.TensorFlowLinearRegressor.bias_` {#TensorFlowLinearRegressor.bias_}

Returns bias of the linear regression.


- - -

#### `tf.contrib.learn.TensorFlowLinearRegressor.fit(X, y, monitor=None, logdir=None)` {#TensorFlowLinearRegressor.fit}

Builds a neural network model given provided `model_fn` and training
data X and y.

Note: called first time constructs the graph and initializers
variables. Consecutives times it will continue training the same model.
This logic follows partial_fit() interface in scikit-learn.

To restart learning, create new estimator.

##### Args:


*  <b>`X`</b>: matrix or tensor of shape [n_samples, n_features...]. Can be
    iterator that returns arrays of features. The training input
    samples for fitting the model.
*  <b>`y`</b>: vector or matrix [n_samples] or [n_samples, n_outputs]. Can be
    iterator that returns array of targets. The training target values
    (class labels in classification, real numbers in regression).
*  <b>`monitor`</b>: Monitor object to print training progress and invoke early stopping
*  <b>`logdir`</b>: the directory to save the log file that can be used for
    optional visualization.

##### Returns:

    Returns self.


- - -

#### `tf.contrib.learn.TensorFlowLinearRegressor.get_params(deep=True)` {#TensorFlowLinearRegressor.get_params}

Get parameters for this estimator.

Parameters
----------
deep: boolean, optional
    If True, will return the parameters for this estimator and
    contained subobjects that are estimators.

Returns
-------
params : mapping of string to any
    Parameter names mapped to their values.


- - -

#### `tf.contrib.learn.TensorFlowLinearRegressor.get_tensor(name)` {#TensorFlowLinearRegressor.get_tensor}

Returns tensor by name.

##### Args:


*  <b>`name`</b>: string, name of the tensor.

##### Returns:

    Tensor.


- - -

#### `tf.contrib.learn.TensorFlowLinearRegressor.get_tensor_value(name)` {#TensorFlowLinearRegressor.get_tensor_value}

Returns value of the tensor give by name.

##### Args:


*  <b>`name`</b>: string, name of the tensor.

##### Returns:

    Numpy array - value of the tensor.


- - -

#### `tf.contrib.learn.TensorFlowLinearRegressor.partial_fit(X, y)` {#TensorFlowLinearRegressor.partial_fit}

Incremental fit on a batch of samples.

This method is expected to be called several times consecutively
on different or the same chunks of the dataset. This either can
implement iterative training or out-of-core/online training.

This is especially useful when the whole dataset is too big to
fit in memory at the same time. Or when model is taking long time
to converge, and you want to split up training into subparts.

##### Args:


*  <b>`X`</b>: matrix or tensor of shape [n_samples, n_features...]. Can be
    iterator that returns arrays of features. The training input
    samples for fitting the model.
*  <b>`y`</b>: vector or matrix [n_samples] or [n_samples, n_outputs]. Can be
    iterator that returns array of targets. The training target values
    (class label in classification, real numbers in regression).

##### Returns:

    Returns self.


- - -

#### `tf.contrib.learn.TensorFlowLinearRegressor.predict(X, axis=1, batch_size=None)` {#TensorFlowLinearRegressor.predict}

Predict class or regression for X.

For a classification model, the predicted class for each sample in X is
returned. For a regression model, the predicted value based on X is
returned.

##### Args:


*  <b>`X`</b>: array-like matrix, [n_samples, n_features...] or iterator.
*  <b>`axis`</b>: Which axis to argmax for classification.
          By default axis 1 (next after batch) is used.
          Use 2 for sequence predictions.
*  <b>`batch_size`</b>: If test set is too big, use batch size to split
                it into mini batches. By default the batch_size member
                variable is used.

##### Returns:


*  <b>`y`</b>: array of shape [n_samples]. The predicted classes or predicted
    value.


- - -

#### `tf.contrib.learn.TensorFlowLinearRegressor.predict_proba(X, batch_size=None)` {#TensorFlowLinearRegressor.predict_proba}

Predict class probability of the input samples X.

##### Args:


*  <b>`X`</b>: array-like matrix, [n_samples, n_features...] or iterator.
*  <b>`batch_size`</b>: If test set is too big, use batch size to split
                it into mini batches. By default the batch_size
                member variable is used.

##### Returns:


*  <b>`y`</b>: array of shape [n_samples, n_classes]. The predicted
    probabilities for each class.


- - -

#### `tf.contrib.learn.TensorFlowLinearRegressor.restore(cls, path, config=None)` {#TensorFlowLinearRegressor.restore}

Restores model from give path.

##### Args:


*  <b>`path`</b>: Path to the checkpoints and other model information.
*  <b>`config`</b>: RunConfig object that controls the configurations of the session,
        e.g. num_cores, gpu_memory_fraction, etc. This is allowed to be reconfigured.

##### Returns:

    Estiamator, object of the subclass of TensorFlowEstimator.


- - -

#### `tf.contrib.learn.TensorFlowLinearRegressor.save(path)` {#TensorFlowLinearRegressor.save}

Saves checkpoints and graph to given path.

##### Args:


*  <b>`path`</b>: Folder to save model to.


- - -

#### `tf.contrib.learn.TensorFlowLinearRegressor.score(X, y, sample_weight=None)` {#TensorFlowLinearRegressor.score}

Returns the coefficient of determination R^2 of the prediction.

The coefficient R^2 is defined as (1 - u/v), where u is the regression
sum of squares ((y_true - y_pred) ** 2).sum() and v is the residual
sum of squares ((y_true - y_true.mean()) ** 2).sum().
Best possible score is 1.0 and it can be negative (because the
model can be arbitrarily worse). A constant model that always
predicts the expected value of y, disregarding the input features,
would get a R^2 score of 0.0.

Parameters
----------
X : array-like, shape = (n_samples, n_features)
    Test samples.

y : array-like, shape = (n_samples) or (n_samples, n_outputs)
    True values for X.

sample_weight : array-like, shape = [n_samples], optional
    Sample weights.

Returns
-------
score : float
    R^2 of self.predict(X) wrt. y.


- - -

#### `tf.contrib.learn.TensorFlowLinearRegressor.set_params(**params)` {#TensorFlowLinearRegressor.set_params}

Set the parameters of this estimator.

The method works on simple estimators as well as on nested objects
(such as pipelines). The former have parameters of the form
``<component>__<parameter>`` so that it's possible to update each
component of a nested object.

Returns
-------
self


- - -

#### `tf.contrib.learn.TensorFlowLinearRegressor.weights_` {#TensorFlowLinearRegressor.weights_}

Returns weights of the linear regression.



- - -

### `class tf.contrib.learn.TensorFlowRNNClassifier` {#TensorFlowRNNClassifier}

TensorFlow RNN Classifier model.

Parameters:
    rnn_size: The size for rnn cell, e.g. size of your word embeddings.
    cell_type: The type of rnn cell, including rnn, gru, and lstm.
    num_layers: The number of layers of the rnn model.
    input_op_fn: Function that will transform the input tensor, such as
                 creating word embeddings, byte list, etc. This takes
                 an argument X for input and returns transformed X.
    bidirectional: boolean, Whether this is a bidirectional rnn.
    sequence_length: If sequence_length is provided, dynamic calculation is performed.
             This saves computational time when unrolling past max sequence length.
    initial_state: An initial state for the RNN. This must be a tensor of appropriate type
                   and shape [batch_size x cell.state_size].
    n_classes: Number of classes in the target.
    batch_size: Mini batch size.
    steps: Number of steps to run over data.
    optimizer: Optimizer name (or class), for example "SGD", "Adam",
               "Adagrad".
    learning_rate: If this is constant float value, no decay function is used.
        Instead, a customized decay function can be passed that accepts
        global_step as parameter and returns a Tensor.
        e.g. exponential decay function:
        def exp_decay(global_step):
            return tf.train.exponential_decay(
                learning_rate=0.1, global_step,
                decay_steps=2, decay_rate=0.001)
    class_weight: None or list of n_classes floats. Weight associated with
                 classes for loss computation. If not given, all classes are suppose to have
                 weight one.
    continue_training: when continue_training is True, once initialized
        model will be continuely trained on every call of fit.
    config: RunConfig object that controls the configurations of the session,
        e.g. num_cores, gpu_memory_fraction, etc.
- - -

#### `tf.contrib.learn.TensorFlowRNNClassifier.__init__(rnn_size, n_classes, cell_type='gru', num_layers=1, input_op_fn=null_input_op_fn, initial_state=None, bidirectional=False, sequence_length=None, batch_size=32, steps=50, optimizer='Adagrad', learning_rate=0.1, class_weight=None, clip_gradients=5.0, continue_training=False, config=None, verbose=1)` {#TensorFlowRNNClassifier.__init__}




- - -

#### `tf.contrib.learn.TensorFlowRNNClassifier.bias_` {#TensorFlowRNNClassifier.bias_}

Returns bias of the rnn layer.


- - -

#### `tf.contrib.learn.TensorFlowRNNClassifier.fit(X, y, monitor=None, logdir=None)` {#TensorFlowRNNClassifier.fit}

Builds a neural network model given provided `model_fn` and training
data X and y.

Note: called first time constructs the graph and initializers
variables. Consecutives times it will continue training the same model.
This logic follows partial_fit() interface in scikit-learn.

To restart learning, create new estimator.

##### Args:


*  <b>`X`</b>: matrix or tensor of shape [n_samples, n_features...]. Can be
    iterator that returns arrays of features. The training input
    samples for fitting the model.
*  <b>`y`</b>: vector or matrix [n_samples] or [n_samples, n_outputs]. Can be
    iterator that returns array of targets. The training target values
    (class labels in classification, real numbers in regression).
*  <b>`monitor`</b>: Monitor object to print training progress and invoke early stopping
*  <b>`logdir`</b>: the directory to save the log file that can be used for
    optional visualization.

##### Returns:

    Returns self.


- - -

#### `tf.contrib.learn.TensorFlowRNNClassifier.get_params(deep=True)` {#TensorFlowRNNClassifier.get_params}

Get parameters for this estimator.

Parameters
----------
deep: boolean, optional
    If True, will return the parameters for this estimator and
    contained subobjects that are estimators.

Returns
-------
params : mapping of string to any
    Parameter names mapped to their values.


- - -

#### `tf.contrib.learn.TensorFlowRNNClassifier.get_tensor(name)` {#TensorFlowRNNClassifier.get_tensor}

Returns tensor by name.

##### Args:


*  <b>`name`</b>: string, name of the tensor.

##### Returns:

    Tensor.


- - -

#### `tf.contrib.learn.TensorFlowRNNClassifier.get_tensor_value(name)` {#TensorFlowRNNClassifier.get_tensor_value}

Returns value of the tensor give by name.

##### Args:


*  <b>`name`</b>: string, name of the tensor.

##### Returns:

    Numpy array - value of the tensor.


- - -

#### `tf.contrib.learn.TensorFlowRNNClassifier.partial_fit(X, y)` {#TensorFlowRNNClassifier.partial_fit}

Incremental fit on a batch of samples.

This method is expected to be called several times consecutively
on different or the same chunks of the dataset. This either can
implement iterative training or out-of-core/online training.

This is especially useful when the whole dataset is too big to
fit in memory at the same time. Or when model is taking long time
to converge, and you want to split up training into subparts.

##### Args:


*  <b>`X`</b>: matrix or tensor of shape [n_samples, n_features...]. Can be
    iterator that returns arrays of features. The training input
    samples for fitting the model.
*  <b>`y`</b>: vector or matrix [n_samples] or [n_samples, n_outputs]. Can be
    iterator that returns array of targets. The training target values
    (class label in classification, real numbers in regression).

##### Returns:

    Returns self.


- - -

#### `tf.contrib.learn.TensorFlowRNNClassifier.predict(X, axis=1, batch_size=None)` {#TensorFlowRNNClassifier.predict}

Predict class or regression for X.

For a classification model, the predicted class for each sample in X is
returned. For a regression model, the predicted value based on X is
returned.

##### Args:


*  <b>`X`</b>: array-like matrix, [n_samples, n_features...] or iterator.
*  <b>`axis`</b>: Which axis to argmax for classification.
          By default axis 1 (next after batch) is used.
          Use 2 for sequence predictions.
*  <b>`batch_size`</b>: If test set is too big, use batch size to split
                it into mini batches. By default the batch_size member
                variable is used.

##### Returns:


*  <b>`y`</b>: array of shape [n_samples]. The predicted classes or predicted
    value.


- - -

#### `tf.contrib.learn.TensorFlowRNNClassifier.predict_proba(X, batch_size=None)` {#TensorFlowRNNClassifier.predict_proba}

Predict class probability of the input samples X.

##### Args:


*  <b>`X`</b>: array-like matrix, [n_samples, n_features...] or iterator.
*  <b>`batch_size`</b>: If test set is too big, use batch size to split
                it into mini batches. By default the batch_size
                member variable is used.

##### Returns:


*  <b>`y`</b>: array of shape [n_samples, n_classes]. The predicted
    probabilities for each class.


- - -

#### `tf.contrib.learn.TensorFlowRNNClassifier.restore(cls, path, config=None)` {#TensorFlowRNNClassifier.restore}

Restores model from give path.

##### Args:


*  <b>`path`</b>: Path to the checkpoints and other model information.
*  <b>`config`</b>: RunConfig object that controls the configurations of the session,
        e.g. num_cores, gpu_memory_fraction, etc. This is allowed to be reconfigured.

##### Returns:

    Estiamator, object of the subclass of TensorFlowEstimator.


- - -

#### `tf.contrib.learn.TensorFlowRNNClassifier.save(path)` {#TensorFlowRNNClassifier.save}

Saves checkpoints and graph to given path.

##### Args:


*  <b>`path`</b>: Folder to save model to.


- - -

#### `tf.contrib.learn.TensorFlowRNNClassifier.score(X, y, sample_weight=None)` {#TensorFlowRNNClassifier.score}

Returns the mean accuracy on the given test data and labels.

In multi-label classification, this is the subset accuracy
which is a harsh metric since you require for each sample that
each label set be correctly predicted.

Parameters
----------
X : array-like, shape = (n_samples, n_features)
    Test samples.

y : array-like, shape = (n_samples) or (n_samples, n_outputs)
    True labels for X.

sample_weight : array-like, shape = [n_samples], optional
    Sample weights.

Returns
-------
score : float
    Mean accuracy of self.predict(X) wrt. y.


- - -

#### `tf.contrib.learn.TensorFlowRNNClassifier.set_params(**params)` {#TensorFlowRNNClassifier.set_params}

Set the parameters of this estimator.

The method works on simple estimators as well as on nested objects
(such as pipelines). The former have parameters of the form
``<component>__<parameter>`` so that it's possible to update each
component of a nested object.

Returns
-------
self


- - -

#### `tf.contrib.learn.TensorFlowRNNClassifier.weights_` {#TensorFlowRNNClassifier.weights_}

Returns weights of the rnn layer.



- - -

### `class tf.contrib.learn.TensorFlowRNNRegressor` {#TensorFlowRNNRegressor}

TensorFlow RNN Regressor model.

Parameters:
    rnn_size: The size for rnn cell, e.g. size of your word embeddings.
    cell_type: The type of rnn cell, including rnn, gru, and lstm.
    num_layers: The number of layers of the rnn model.
    input_op_fn: Function that will transform the input tensor, such as
                 creating word embeddings, byte list, etc. This takes
                 an argument X for input and returns transformed X.
    bidirectional: boolean, Whether this is a bidirectional rnn.
    sequence_length: If sequence_length is provided, dynamic calculation is performed.
             This saves computational time when unrolling past max sequence length.
    initial_state: An initial state for the RNN. This must be a tensor of appropriate type
                   and shape [batch_size x cell.state_size].
    batch_size: Mini batch size.
    steps: Number of steps to run over data.
    optimizer: Optimizer name (or class), for example "SGD", "Adam",
               "Adagrad".
    learning_rate: If this is constant float value, no decay function is used.
        Instead, a customized decay function can be passed that accepts
        global_step as parameter and returns a Tensor.
        e.g. exponential decay function:
        def exp_decay(global_step):
            return tf.train.exponential_decay(
                learning_rate=0.1, global_step,
                decay_steps=2, decay_rate=0.001)
    continue_training: when continue_training is True, once initialized
        model will be continuely trained on every call of fit.
    config: RunConfig object that controls the configurations of the session,
        e.g. num_cores, gpu_memory_fraction, etc.
    verbose: Controls the verbosity, possible values:
             0: the algorithm and debug information is muted.
             1: trainer prints the progress.
             2: log device placement is printed.
- - -

#### `tf.contrib.learn.TensorFlowRNNRegressor.__init__(rnn_size, cell_type='gru', num_layers=1, input_op_fn=null_input_op_fn, initial_state=None, bidirectional=False, sequence_length=None, n_classes=0, batch_size=32, steps=50, optimizer='Adagrad', learning_rate=0.1, clip_gradients=5.0, continue_training=False, config=None, verbose=1)` {#TensorFlowRNNRegressor.__init__}




- - -

#### `tf.contrib.learn.TensorFlowRNNRegressor.bias_` {#TensorFlowRNNRegressor.bias_}

Returns bias of the rnn layer.


- - -

#### `tf.contrib.learn.TensorFlowRNNRegressor.fit(X, y, monitor=None, logdir=None)` {#TensorFlowRNNRegressor.fit}

Builds a neural network model given provided `model_fn` and training
data X and y.

Note: called first time constructs the graph and initializers
variables. Consecutives times it will continue training the same model.
This logic follows partial_fit() interface in scikit-learn.

To restart learning, create new estimator.

##### Args:


*  <b>`X`</b>: matrix or tensor of shape [n_samples, n_features...]. Can be
    iterator that returns arrays of features. The training input
    samples for fitting the model.
*  <b>`y`</b>: vector or matrix [n_samples] or [n_samples, n_outputs]. Can be
    iterator that returns array of targets. The training target values
    (class labels in classification, real numbers in regression).
*  <b>`monitor`</b>: Monitor object to print training progress and invoke early stopping
*  <b>`logdir`</b>: the directory to save the log file that can be used for
    optional visualization.

##### Returns:

    Returns self.


- - -

#### `tf.contrib.learn.TensorFlowRNNRegressor.get_params(deep=True)` {#TensorFlowRNNRegressor.get_params}

Get parameters for this estimator.

Parameters
----------
deep: boolean, optional
    If True, will return the parameters for this estimator and
    contained subobjects that are estimators.

Returns
-------
params : mapping of string to any
    Parameter names mapped to their values.


- - -

#### `tf.contrib.learn.TensorFlowRNNRegressor.get_tensor(name)` {#TensorFlowRNNRegressor.get_tensor}

Returns tensor by name.

##### Args:


*  <b>`name`</b>: string, name of the tensor.

##### Returns:

    Tensor.


- - -

#### `tf.contrib.learn.TensorFlowRNNRegressor.get_tensor_value(name)` {#TensorFlowRNNRegressor.get_tensor_value}

Returns value of the tensor give by name.

##### Args:


*  <b>`name`</b>: string, name of the tensor.

##### Returns:

    Numpy array - value of the tensor.


- - -

#### `tf.contrib.learn.TensorFlowRNNRegressor.partial_fit(X, y)` {#TensorFlowRNNRegressor.partial_fit}

Incremental fit on a batch of samples.

This method is expected to be called several times consecutively
on different or the same chunks of the dataset. This either can
implement iterative training or out-of-core/online training.

This is especially useful when the whole dataset is too big to
fit in memory at the same time. Or when model is taking long time
to converge, and you want to split up training into subparts.

##### Args:


*  <b>`X`</b>: matrix or tensor of shape [n_samples, n_features...]. Can be
    iterator that returns arrays of features. The training input
    samples for fitting the model.
*  <b>`y`</b>: vector or matrix [n_samples] or [n_samples, n_outputs]. Can be
    iterator that returns array of targets. The training target values
    (class label in classification, real numbers in regression).

##### Returns:

    Returns self.


- - -

#### `tf.contrib.learn.TensorFlowRNNRegressor.predict(X, axis=1, batch_size=None)` {#TensorFlowRNNRegressor.predict}

Predict class or regression for X.

For a classification model, the predicted class for each sample in X is
returned. For a regression model, the predicted value based on X is
returned.

##### Args:


*  <b>`X`</b>: array-like matrix, [n_samples, n_features...] or iterator.
*  <b>`axis`</b>: Which axis to argmax for classification.
          By default axis 1 (next after batch) is used.
          Use 2 for sequence predictions.
*  <b>`batch_size`</b>: If test set is too big, use batch size to split
                it into mini batches. By default the batch_size member
                variable is used.

##### Returns:


*  <b>`y`</b>: array of shape [n_samples]. The predicted classes or predicted
    value.


- - -

#### `tf.contrib.learn.TensorFlowRNNRegressor.predict_proba(X, batch_size=None)` {#TensorFlowRNNRegressor.predict_proba}

Predict class probability of the input samples X.

##### Args:


*  <b>`X`</b>: array-like matrix, [n_samples, n_features...] or iterator.
*  <b>`batch_size`</b>: If test set is too big, use batch size to split
                it into mini batches. By default the batch_size
                member variable is used.

##### Returns:


*  <b>`y`</b>: array of shape [n_samples, n_classes]. The predicted
    probabilities for each class.


- - -

#### `tf.contrib.learn.TensorFlowRNNRegressor.restore(cls, path, config=None)` {#TensorFlowRNNRegressor.restore}

Restores model from give path.

##### Args:


*  <b>`path`</b>: Path to the checkpoints and other model information.
*  <b>`config`</b>: RunConfig object that controls the configurations of the session,
        e.g. num_cores, gpu_memory_fraction, etc. This is allowed to be reconfigured.

##### Returns:

    Estiamator, object of the subclass of TensorFlowEstimator.


- - -

#### `tf.contrib.learn.TensorFlowRNNRegressor.save(path)` {#TensorFlowRNNRegressor.save}

Saves checkpoints and graph to given path.

##### Args:


*  <b>`path`</b>: Folder to save model to.


- - -

#### `tf.contrib.learn.TensorFlowRNNRegressor.score(X, y, sample_weight=None)` {#TensorFlowRNNRegressor.score}

Returns the coefficient of determination R^2 of the prediction.

The coefficient R^2 is defined as (1 - u/v), where u is the regression
sum of squares ((y_true - y_pred) ** 2).sum() and v is the residual
sum of squares ((y_true - y_true.mean()) ** 2).sum().
Best possible score is 1.0 and it can be negative (because the
model can be arbitrarily worse). A constant model that always
predicts the expected value of y, disregarding the input features,
would get a R^2 score of 0.0.

Parameters
----------
X : array-like, shape = (n_samples, n_features)
    Test samples.

y : array-like, shape = (n_samples) or (n_samples, n_outputs)
    True values for X.

sample_weight : array-like, shape = [n_samples], optional
    Sample weights.

Returns
-------
score : float
    R^2 of self.predict(X) wrt. y.


- - -

#### `tf.contrib.learn.TensorFlowRNNRegressor.set_params(**params)` {#TensorFlowRNNRegressor.set_params}

Set the parameters of this estimator.

The method works on simple estimators as well as on nested objects
(such as pipelines). The former have parameters of the form
``<component>__<parameter>`` so that it's possible to update each
component of a nested object.

Returns
-------
self


- - -

#### `tf.contrib.learn.TensorFlowRNNRegressor.weights_` {#TensorFlowRNNRegressor.weights_}

Returns weights of the rnn layer.



- - -

### `class tf.contrib.learn.TensorFlowRegressor` {#TensorFlowRegressor}

TensorFlow Linear Regression model.
- - -

#### `tf.contrib.learn.TensorFlowRegressor.__init__(n_classes=0, batch_size=32, steps=200, optimizer='Adagrad', learning_rate=0.1, clip_gradients=5.0, continue_training=False, config=None, verbose=1)` {#TensorFlowRegressor.__init__}




- - -

#### `tf.contrib.learn.TensorFlowRegressor.bias_` {#TensorFlowRegressor.bias_}

Returns bias of the linear regression.


- - -

#### `tf.contrib.learn.TensorFlowRegressor.fit(X, y, monitor=None, logdir=None)` {#TensorFlowRegressor.fit}

Builds a neural network model given provided `model_fn` and training
data X and y.

Note: called first time constructs the graph and initializers
variables. Consecutives times it will continue training the same model.
This logic follows partial_fit() interface in scikit-learn.

To restart learning, create new estimator.

##### Args:


*  <b>`X`</b>: matrix or tensor of shape [n_samples, n_features...]. Can be
    iterator that returns arrays of features. The training input
    samples for fitting the model.
*  <b>`y`</b>: vector or matrix [n_samples] or [n_samples, n_outputs]. Can be
    iterator that returns array of targets. The training target values
    (class labels in classification, real numbers in regression).
*  <b>`monitor`</b>: Monitor object to print training progress and invoke early stopping
*  <b>`logdir`</b>: the directory to save the log file that can be used for
    optional visualization.

##### Returns:

    Returns self.


- - -

#### `tf.contrib.learn.TensorFlowRegressor.get_params(deep=True)` {#TensorFlowRegressor.get_params}

Get parameters for this estimator.

Parameters
----------
deep: boolean, optional
    If True, will return the parameters for this estimator and
    contained subobjects that are estimators.

Returns
-------
params : mapping of string to any
    Parameter names mapped to their values.


- - -

#### `tf.contrib.learn.TensorFlowRegressor.get_tensor(name)` {#TensorFlowRegressor.get_tensor}

Returns tensor by name.

##### Args:


*  <b>`name`</b>: string, name of the tensor.

##### Returns:

    Tensor.


- - -

#### `tf.contrib.learn.TensorFlowRegressor.get_tensor_value(name)` {#TensorFlowRegressor.get_tensor_value}

Returns value of the tensor give by name.

##### Args:


*  <b>`name`</b>: string, name of the tensor.

##### Returns:

    Numpy array - value of the tensor.


- - -

#### `tf.contrib.learn.TensorFlowRegressor.partial_fit(X, y)` {#TensorFlowRegressor.partial_fit}

Incremental fit on a batch of samples.

This method is expected to be called several times consecutively
on different or the same chunks of the dataset. This either can
implement iterative training or out-of-core/online training.

This is especially useful when the whole dataset is too big to
fit in memory at the same time. Or when model is taking long time
to converge, and you want to split up training into subparts.

##### Args:


*  <b>`X`</b>: matrix or tensor of shape [n_samples, n_features...]. Can be
    iterator that returns arrays of features. The training input
    samples for fitting the model.
*  <b>`y`</b>: vector or matrix [n_samples] or [n_samples, n_outputs]. Can be
    iterator that returns array of targets. The training target values
    (class label in classification, real numbers in regression).

##### Returns:

    Returns self.


- - -

#### `tf.contrib.learn.TensorFlowRegressor.predict(X, axis=1, batch_size=None)` {#TensorFlowRegressor.predict}

Predict class or regression for X.

For a classification model, the predicted class for each sample in X is
returned. For a regression model, the predicted value based on X is
returned.

##### Args:


*  <b>`X`</b>: array-like matrix, [n_samples, n_features...] or iterator.
*  <b>`axis`</b>: Which axis to argmax for classification.
          By default axis 1 (next after batch) is used.
          Use 2 for sequence predictions.
*  <b>`batch_size`</b>: If test set is too big, use batch size to split
                it into mini batches. By default the batch_size member
                variable is used.

##### Returns:


*  <b>`y`</b>: array of shape [n_samples]. The predicted classes or predicted
    value.


- - -

#### `tf.contrib.learn.TensorFlowRegressor.predict_proba(X, batch_size=None)` {#TensorFlowRegressor.predict_proba}

Predict class probability of the input samples X.

##### Args:


*  <b>`X`</b>: array-like matrix, [n_samples, n_features...] or iterator.
*  <b>`batch_size`</b>: If test set is too big, use batch size to split
                it into mini batches. By default the batch_size
                member variable is used.

##### Returns:


*  <b>`y`</b>: array of shape [n_samples, n_classes]. The predicted
    probabilities for each class.


- - -

#### `tf.contrib.learn.TensorFlowRegressor.restore(cls, path, config=None)` {#TensorFlowRegressor.restore}

Restores model from give path.

##### Args:


*  <b>`path`</b>: Path to the checkpoints and other model information.
*  <b>`config`</b>: RunConfig object that controls the configurations of the session,
        e.g. num_cores, gpu_memory_fraction, etc. This is allowed to be reconfigured.

##### Returns:

    Estiamator, object of the subclass of TensorFlowEstimator.


- - -

#### `tf.contrib.learn.TensorFlowRegressor.save(path)` {#TensorFlowRegressor.save}

Saves checkpoints and graph to given path.

##### Args:


*  <b>`path`</b>: Folder to save model to.


- - -

#### `tf.contrib.learn.TensorFlowRegressor.score(X, y, sample_weight=None)` {#TensorFlowRegressor.score}

Returns the coefficient of determination R^2 of the prediction.

The coefficient R^2 is defined as (1 - u/v), where u is the regression
sum of squares ((y_true - y_pred) ** 2).sum() and v is the residual
sum of squares ((y_true - y_true.mean()) ** 2).sum().
Best possible score is 1.0 and it can be negative (because the
model can be arbitrarily worse). A constant model that always
predicts the expected value of y, disregarding the input features,
would get a R^2 score of 0.0.

Parameters
----------
X : array-like, shape = (n_samples, n_features)
    Test samples.

y : array-like, shape = (n_samples) or (n_samples, n_outputs)
    True values for X.

sample_weight : array-like, shape = [n_samples], optional
    Sample weights.

Returns
-------
score : float
    R^2 of self.predict(X) wrt. y.


- - -

#### `tf.contrib.learn.TensorFlowRegressor.set_params(**params)` {#TensorFlowRegressor.set_params}

Set the parameters of this estimator.

The method works on simple estimators as well as on nested objects
(such as pipelines). The former have parameters of the form
``<component>__<parameter>`` so that it's possible to update each
component of a nested object.

Returns
-------
self


- - -

#### `tf.contrib.learn.TensorFlowRegressor.weights_` {#TensorFlowRegressor.weights_}

Returns weights of the linear regression.



- - -

### `tf.contrib.learn.extract_dask_data(data)` {#extract_dask_data}

Extract data from dask.Series or dask.DataFrame for predictors


- - -

### `tf.contrib.learn.extract_dask_labels(labels)` {#extract_dask_labels}

Extract data from dask.Series for labels


- - -

### `tf.contrib.learn.extract_pandas_data(data)` {#extract_pandas_data}

Extract data from pandas.DataFrame for predictors


- - -

### `tf.contrib.learn.extract_pandas_labels(labels)` {#extract_pandas_labels}

Extract data from pandas.DataFrame for labels


- - -

### `tf.contrib.learn.extract_pandas_matrix(data)` {#extract_pandas_matrix}

Extracts numpy matrix from pandas DataFrame.


