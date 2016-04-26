# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


class BatchMatrixDiagTest(tf.test.TestCase):
  _use_gpu = False

  def testVector(self):
    with self.test_session(use_gpu=self._use_gpu):
      v = np.array([1.0, 2.0, 3.0])
      mat = np.diag(v)
      v_diag = tf.batch_matrix_diag(v)
      self.assertEqual((3, 3), v_diag.get_shape())
      self.assertAllEqual(v_diag.eval(), mat)

  def testBatchVector(self):
    with self.test_session(use_gpu=self._use_gpu):
      v_batch = np.array([[1.0, 2.0, 3.0],
                          [4.0, 5.0, 6.0]])
      mat_batch = np.array(
          [[[1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 3.0]],
           [[4.0, 0.0, 0.0],
            [0.0, 5.0, 0.0],
            [0.0, 0.0, 6.0]]])
      v_batch_diag = tf.batch_matrix_diag(v_batch)
      self.assertEqual((2, 3, 3), v_batch_diag.get_shape())
      self.assertAllEqual(v_batch_diag.eval(), mat_batch)

  def testInvalidShape(self):
    with self.assertRaisesRegexp(ValueError, "must have rank at least 1"):
      tf.batch_matrix_diag(0)

  def testInvalidShapeAtEval(self):
    with self.test_session(use_gpu=self._use_gpu):
      v = tf.placeholder(dtype=tf.float32)
      with self.assertRaisesOpError("input must be at least 1-dim"):
        tf.batch_matrix_diag(v).eval(feed_dict={v: 0.0})

  def testGrad(self):
    shapes = ((3,), (7, 4))
    with self.test_session(use_gpu=self._use_gpu):
      for shape in shapes:
        x = tf.constant(np.random.rand(*shape), np.float32)
        y = tf.batch_matrix_diag(x)
        error = tf.test.compute_gradient_error(x, x.get_shape().as_list(),
                                               y, y.get_shape().as_list())
        self.assertLess(error, 1e-4)


class BatchMatrixDiagGpuTest(BatchMatrixDiagTest):
  _use_gpu = True


class BatchMatrixDiagPartTest(tf.test.TestCase):
  _use_gpu = False

  def testMatrix(self):
    with self.test_session(use_gpu=self._use_gpu):
      v = np.array([1.0, 2.0, 3.0])
      mat = np.diag(v)
      mat_diag = tf.batch_matrix_diag_part(mat)
      self.assertEqual((3,), mat_diag.get_shape())
      self.assertAllEqual(mat_diag.eval(), v)

  def testBatchMatrix(self):
    with self.test_session(use_gpu=self._use_gpu):
      v_batch = np.array([[1.0, 2.0, 3.0],
                          [4.0, 5.0, 6.0]])
      mat_batch = np.array(
          [[[1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 3.0]],
           [[4.0, 0.0, 0.0],
            [0.0, 5.0, 0.0],
            [0.0, 0.0, 6.0]]])
      self.assertEqual(mat_batch.shape, (2, 3, 3))
      mat_batch_diag = tf.batch_matrix_diag_part(mat_batch)
      self.assertEqual((2, 3), mat_batch_diag.get_shape())
      self.assertAllEqual(mat_batch_diag.eval(), v_batch)

  def testInvalidShape(self):
    with self.assertRaisesRegexp(ValueError, "must have rank at least 2"):
      tf.batch_matrix_diag_part(0)
    with self.assertRaisesRegexp(ValueError, r"Dimensions .* not compatible"):
      tf.batch_matrix_diag_part([[0, 1], [1, 0], [0, 0]])

  def testInvalidShapeAtEval(self):
    with self.test_session(use_gpu=self._use_gpu):
      v = tf.placeholder(dtype=tf.float32)
      with self.assertRaisesOpError("input must be at least 2-dim"):
        tf.batch_matrix_diag_part(v).eval(feed_dict={v: 0.0})
      with self.assertRaisesOpError("last two dimensions must be equal"):
        tf.batch_matrix_diag_part(v).eval(
            feed_dict={v: [[0, 1], [1, 0], [0, 0]]})

  def testGrad(self):
    shapes = ((3, 3), (5, 3, 3))
    with self.test_session(use_gpu=self._use_gpu):
      for shape in shapes:
        x = tf.constant(np.random.rand(*shape), dtype=np.float32)
        y = tf.batch_matrix_diag_part(x)
        error = tf.test.compute_gradient_error(x, x.get_shape().as_list(),
                                               y, y.get_shape().as_list())
        self.assertLess(error, 1e-4)


class BatchMatrixDiagPartGpuTest(BatchMatrixDiagPartTest):
  _use_gpu = True


class DiagTest(tf.test.TestCase):

  def diagOp(self, diag, dtype, expected_ans, use_gpu=False):
    with self.test_session(use_gpu=use_gpu):
      tf_ans = tf.diag(tf.convert_to_tensor(diag.astype(dtype)))
      out = tf_ans.eval()
      tf_ans_inv = tf.diag_part(expected_ans)
      inv_out = tf_ans_inv.eval()
    self.assertAllClose(out, expected_ans)
    self.assertAllClose(inv_out, diag)
    self.assertShapeEqual(expected_ans, tf_ans)
    self.assertShapeEqual(diag, tf_ans_inv)

  def testEmptyTensor(self):
    x = np.array([])
    expected_ans = np.empty([0, 0])
    self.diagOp(x, np.int32, expected_ans)

  def testRankOneIntTensor(self):
    x = np.array([1, 2, 3])
    expected_ans = np.array(
        [[1, 0, 0],
         [0, 2, 0],
         [0, 0, 3]])
    self.diagOp(x, np.int32, expected_ans)
    self.diagOp(x, np.int64, expected_ans)

  def testRankOneFloatTensor(self):
    x = np.array([1.1, 2.2, 3.3])
    expected_ans = np.array(
        [[1.1, 0, 0],
         [0, 2.2, 0],
         [0, 0, 3.3]])
    self.diagOp(x, np.float32, expected_ans)
    self.diagOp(x, np.float64, expected_ans)

  def testRankTwoIntTensor(self):
    x = np.array([[1, 2, 3], [4, 5, 6]])
    expected_ans = np.array(
        [[[[1, 0, 0], [0, 0, 0]],
          [[0, 2, 0], [0, 0, 0]],
          [[0, 0, 3], [0, 0, 0]]],
         [[[0, 0, 0], [4, 0, 0]],
          [[0, 0, 0], [0, 5, 0]],
          [[0, 0, 0], [0, 0, 6]]]])
    self.diagOp(x, np.int32, expected_ans)
    self.diagOp(x, np.int64, expected_ans)

  def testRankTwoFloatTensor(self):
    x = np.array([[1.1, 2.2, 3.3], [4.4, 5.5, 6.6]])
    expected_ans = np.array(
        [[[[1.1, 0, 0], [0, 0, 0]],
          [[0, 2.2, 0], [0, 0, 0]],
          [[0, 0, 3.3], [0, 0, 0]]],
         [[[0, 0, 0], [4.4, 0, 0]],
          [[0, 0, 0], [0, 5.5, 0]],
          [[0, 0, 0], [0, 0, 6.6]]]])
    self.diagOp(x, np.float32, expected_ans)
    self.diagOp(x, np.float64, expected_ans)

  def testRankThreeFloatTensor(self):
    x = np.array([[[1.1, 2.2], [3.3, 4.4]],
                  [[5.5, 6.6], [7.7, 8.8]]])
    expected_ans = np.array(
        [[[[[[1.1, 0], [0, 0]], [[0, 0], [0, 0]]],
           [[[0, 2.2], [0, 0]], [[0, 0], [0, 0]]]],
          [[[[0, 0], [3.3, 0]], [[0, 0], [0, 0]]],
           [[[0, 0], [0, 4.4]], [[0, 0], [0, 0]]]]],
         [[[[[0, 0], [0, 0]], [[5.5, 0], [0, 0]]],
           [[[0, 0], [0, 0]], [[0, 6.6], [0, 0]]]],
          [[[[0, 0], [0, 0]], [[0, 0], [7.7, 0]]],
           [[[0, 0], [0, 0]], [[0, 0], [0, 8.8]]]]]])
    self.diagOp(x, np.float32, expected_ans)
    self.diagOp(x, np.float64, expected_ans)


class DiagPartOpTest(tf.test.TestCase):

  def setUp(self):
    np.random.seed(0)

  def diagPartOp(self, tensor, dtype, expected_ans, use_gpu=False):
    with self.test_session(use_gpu=use_gpu):
      tensor = tf.convert_to_tensor(tensor.astype(dtype))
      tf_ans_inv = tf.diag_part(tensor)
      inv_out = tf_ans_inv.eval()
    self.assertAllClose(inv_out, expected_ans)
    self.assertShapeEqual(expected_ans, tf_ans_inv)

  def testRankTwoFloatTensor(self):
    x = np.random.rand(3, 3)
    i = np.arange(3)
    expected_ans = x[i, i]
    self.diagPartOp(x, np.float32, expected_ans)
    self.diagPartOp(x, np.float64, expected_ans)

  def testRankFourFloatTensorUnknownShape(self):
    x = np.random.rand(3, 3)
    i = np.arange(3)
    expected_ans = x[i, i]
    for shape in None, (None, 3), (3, None):
      with self.test_session(use_gpu=False):
        t = tf.convert_to_tensor(x.astype(np.float32))
        t.set_shape(shape)
        tf_ans = tf.diag_part(t)
        out = tf_ans.eval()
      self.assertAllClose(out, expected_ans)
      self.assertShapeEqual(expected_ans, tf_ans)

  def testRankFourFloatTensor(self):
    x = np.random.rand(2, 3, 2, 3)
    i = np.arange(2)[:, None]
    j = np.arange(3)
    expected_ans = x[i, j, i, j]
    self.diagPartOp(x, np.float32, expected_ans)
    self.diagPartOp(x, np.float64, expected_ans)

  def testRankSixFloatTensor(self):
    x = np.random.rand(2, 2, 2, 2, 2, 2)
    i = np.arange(2)[:, None, None]
    j = np.arange(2)[:, None]
    k = np.arange(2)
    expected_ans = x[i, j, k, i, j, k]
    self.diagPartOp(x, np.float32, expected_ans)
    self.diagPartOp(x, np.float64, expected_ans)

  def testOddRank(self):
    w = np.random.rand(2)
    x = np.random.rand(2, 2, 2)
    self.assertRaises(ValueError, self.diagPartOp, w, np.float32, 0)
    self.assertRaises(ValueError, self.diagPartOp, x, np.float32, 0)

  def testUnevenDimensions(self):
    w = np.random.rand(2, 5)
    x = np.random.rand(2, 1, 2, 3)
    self.assertRaises(ValueError, self.diagPartOp, w, np.float32, 0)
    self.assertRaises(ValueError, self.diagPartOp, x, np.float32, 0)


class DiagGradOpTest(tf.test.TestCase):

  def testDiagGrad(self):
    np.random.seed(0)
    shapes = ((3,), (3,3), (3,3,3))
    dtypes = (tf.float32, tf.float64)
    with self.test_session(use_gpu=False):
      errors = []
      for shape in shapes:
        for dtype in dtypes:
          x1 = tf.constant(np.random.rand(*shape), dtype=dtype)
          y = tf.diag(x1)
          error = tf.test.compute_gradient_error(x1, x1.get_shape().as_list(),
                                                 y, y.get_shape().as_list())
          tf.logging.info("error = %f", error)
          self.assertLess(error, 1e-4)


class DiagGradPartOpTest(tf.test.TestCase):

  def testDiagPartGrad(self):
    np.random.seed(0)
    shapes = ((3,3), (3,3,3,3))
    dtypes = (tf.float32, tf.float64)
    with self.test_session(use_gpu=False):
      errors = []
      for shape in shapes:
        for dtype in dtypes:
          x1 = tf.constant(np.random.rand(*shape), dtype=dtype)
          y = tf.diag_part(x1)
          error = tf.test.compute_gradient_error(x1, x1.get_shape().as_list(),
                                                 y, y.get_shape().as_list())
          tf.logging.info("error = %f", error)
          self.assertLess(error, 1e-4)

if __name__ == "__main__":
  tf.test.main()
