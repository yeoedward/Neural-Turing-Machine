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

"""Functional tests for tensor_util."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import state_ops


class TensorUtilTest(tf.test.TestCase):

  def testFloat(self):
    t = tensor_util.make_tensor_proto(10.0)
    self.assertProtoEquals("""
      dtype: DT_FLOAT
      tensor_shape {}
      float_val: 10.0
      """, t)
    a = tensor_util.MakeNdarray(t)
    self.assertEquals(np.float32, a.dtype)
    self.assertAllClose(np.array(10.0, dtype=np.float32), a)

  def testFloatN(self):
    t = tensor_util.make_tensor_proto([10.0, 20.0, 30.0])
    self.assertProtoEquals("""
      dtype: DT_FLOAT
      tensor_shape { dim { size: 3 } }
      tensor_content: "\000\000 A\000\000\240A\000\000\360A"
      """, t)
    a = tensor_util.MakeNdarray(t)
    self.assertEquals(np.float32, a.dtype)
    self.assertAllClose(np.array([10.0, 20.0, 30.0], dtype=np.float32), a)

  def testFloatTyped(self):
    t = tensor_util.make_tensor_proto([10.0, 20.0, 30.0], dtype=tf.float32)
    self.assertProtoEquals("""
      dtype: DT_FLOAT
      tensor_shape { dim { size: 3 } }
      tensor_content: "\000\000 A\000\000\240A\000\000\360A"
      """, t)
    a = tensor_util.MakeNdarray(t)
    self.assertEquals(np.float32, a.dtype)
    self.assertAllClose(np.array([10.0, 20.0, 30.0], dtype=np.float32), a)

  def testFloatTypeCoerce(self):
    t = tensor_util.make_tensor_proto([10, 20, 30], dtype=tf.float32)
    self.assertProtoEquals("""
      dtype: DT_FLOAT
      tensor_shape { dim { size: 3 } }
      tensor_content: "\000\000 A\000\000\240A\000\000\360A"
      """, t)
    a = tensor_util.MakeNdarray(t)
    self.assertEquals(np.float32, a.dtype)
    self.assertAllClose(np.array([10.0, 20.0, 30.0], dtype=np.float32), a)

  def testFloatTypeCoerceNdarray(self):
    arr = np.asarray([10, 20, 30], dtype="int")
    t = tensor_util.make_tensor_proto(arr, dtype=tf.float32)
    self.assertProtoEquals("""
      dtype: DT_FLOAT
      tensor_shape { dim { size: 3 } }
      tensor_content: "\000\000 A\000\000\240A\000\000\360A"
      """, t)
    a = tensor_util.MakeNdarray(t)
    self.assertEquals(np.float32, a.dtype)
    self.assertAllClose(np.array([10.0, 20.0, 30.0], dtype=np.float32), a)

  def testFloatSizes(self):
    t = tensor_util.make_tensor_proto([10.0, 20.0, 30.0], shape=[1, 3])
    self.assertProtoEquals("""
      dtype: DT_FLOAT
      tensor_shape { dim { size: 1 } dim { size: 3 } }
      tensor_content: "\000\000 A\000\000\240A\000\000\360A"
      """, t)
    a = tensor_util.MakeNdarray(t)
    self.assertEquals(np.float32, a.dtype)
    self.assertAllClose(np.array([[10.0, 20.0, 30.0]], dtype=np.float32), a)

  def testFloatSizes2(self):
    t = tensor_util.make_tensor_proto([10.0, 20.0, 30.0], shape=[3, 1])
    self.assertProtoEquals("""
      dtype: DT_FLOAT
      tensor_shape { dim { size: 3 } dim { size: 1 } }
      tensor_content: "\000\000 A\000\000\240A\000\000\360A"
      """, t)
    a = tensor_util.MakeNdarray(t)
    self.assertEquals(np.float32, a.dtype)
    self.assertAllClose(np.array([[10.0], [20.0], [30.0]], dtype=np.float32),
                        a)

  def testFloatSizesLessValues(self):
    t = tensor_util.make_tensor_proto(10.0, shape=[1, 3])
    self.assertProtoEquals("""
      dtype: DT_FLOAT
      tensor_shape { dim { size: 1 } dim { size: 3 } }
      float_val: 10.0
      """, t)
    # No conversion to Ndarray for this one: not enough values.

  def testFloatNpArrayFloat64(self):
    t = tensor_util.make_tensor_proto(
        np.array([[10.0, 20.0, 30.0]], dtype=np.float64))
    self.assertProtoEquals("""
      dtype: DT_DOUBLE
      tensor_shape { dim { size: 1 } dim { size: 3 } }
      tensor_content: "\000\000\000\000\000\000$@\000\000\000\000\000\0004@\000\000\000\000\000\000>@"
      """, t)
    a = tensor_util.MakeNdarray(t)
    self.assertEquals(np.float64, a.dtype)
    self.assertAllClose(np.array([[10.0, 20.0, 30.0]], dtype=np.float64),
                        tensor_util.MakeNdarray(t))

  def testFloatTypesWithImplicitRepeat(self):
    for dtype, nptype in [
        (tf.float32, np.float32), (tf.float64, np.float64)]:
      t = tensor_util.make_tensor_proto([10.0], shape=[3, 4], dtype=dtype)
      a = tensor_util.MakeNdarray(t)
      self.assertAllClose(np.array([[10.0, 10.0, 10.0, 10.0],
                                    [10.0, 10.0, 10.0, 10.0],
                                    [10.0, 10.0, 10.0, 10.0]], dtype=nptype), a)

  def testInt(self):
    t = tensor_util.make_tensor_proto(10)
    self.assertProtoEquals("""
      dtype: DT_INT32
      tensor_shape {}
      int_val: 10
      """, t)
    a = tensor_util.MakeNdarray(t)
    self.assertEquals(np.int32, a.dtype)
    self.assertAllClose(np.array(10, dtype=np.int32), a)

  def testIntNDefaultType(self):
    t = tensor_util.make_tensor_proto([10, 20, 30, 40], shape=[2, 2])
    self.assertProtoEquals("""
      dtype: DT_INT32
      tensor_shape { dim { size: 2 } dim { size: 2 } }
      tensor_content: "\\n\000\000\000\024\000\000\000\036\000\000\000(\000\000\000"
      """, t)
    a = tensor_util.MakeNdarray(t)
    self.assertEquals(np.int32, a.dtype)
    self.assertAllClose(np.array([[10, 20], [30, 40]], dtype=np.int32), a)

  def testIntTypes(self):
    for dtype, nptype in [
        (tf.int32, np.int32),
        (tf.uint8, np.uint8),
        (tf.uint16, np.uint16),
        (tf.int16, np.int16),
        (tf.int8, np.int8)]:
      # Test with array.
      t = tensor_util.make_tensor_proto([10, 20, 30], dtype=dtype)
      self.assertEquals(dtype, t.dtype)
      self.assertProtoEquals("dim { size: 3 }", t.tensor_shape)
      a = tensor_util.MakeNdarray(t)
      self.assertEquals(nptype, a.dtype)
      self.assertAllClose(np.array([10, 20, 30], dtype=nptype), a)
      # Test with ndarray.
      t = tensor_util.make_tensor_proto(np.array([10, 20, 30], dtype=nptype))
      self.assertEquals(dtype, t.dtype)
      self.assertProtoEquals("dim { size: 3 }", t.tensor_shape)
      a = tensor_util.MakeNdarray(t)
      self.assertEquals(nptype, a.dtype)
      self.assertAllClose(np.array([10, 20, 30], dtype=nptype), a)

  def testIntTypesWithImplicitRepeat(self):
    for dtype, nptype in [
        (tf.int64, np.int64),
        (tf.int32, np.int32),
        (tf.uint8, np.uint8),
        (tf.uint16, np.uint16),
        (tf.int16, np.int16),
        (tf.int8, np.int8)]:
      t = tensor_util.make_tensor_proto([10], shape=[3, 4], dtype=dtype)
      a = tensor_util.MakeNdarray(t)
      self.assertAllEqual(np.array([[10, 10, 10, 10],
                                    [10, 10, 10, 10],
                                    [10, 10, 10, 10]], dtype=nptype), a)

  def testLong(self):
    t = tensor_util.make_tensor_proto(10, dtype=tf.int64)
    self.assertProtoEquals("""
      dtype: DT_INT64
      tensor_shape {}
      int64_val: 10
      """, t)
    a = tensor_util.MakeNdarray(t)
    self.assertEquals(np.int64, a.dtype)
    self.assertAllClose(np.array(10, dtype=np.int64), a)

  def testLongN(self):
    t = tensor_util.make_tensor_proto([10, 20, 30], shape=[1, 3],
                                      dtype=tf.int64)
    self.assertProtoEquals("""
      dtype: DT_INT64
      tensor_shape { dim { size: 1 } dim { size: 3 } }
      tensor_content: "\\n\000\000\000\000\000\000\000\024\000\000\000\000\000\000\000\036\000\000\000\000\000\000\000"
      """, t)
    a = tensor_util.MakeNdarray(t)
    self.assertEquals(np.int64, a.dtype)
    self.assertAllClose(np.array([[10, 20, 30]], dtype=np.int64), a)

  def testLongNpArray(self):
    t = tensor_util.make_tensor_proto(np.array([10, 20, 30]))
    self.assertProtoEquals("""
      dtype: DT_INT64
      tensor_shape { dim { size: 3 } }
      tensor_content: "\\n\000\000\000\000\000\000\000\024\000\000\000\000\000\000\000\036\000\000\000\000\000\000\000"
      """, t)
    a = tensor_util.MakeNdarray(t)
    self.assertEquals(np.int64, a.dtype)
    self.assertAllClose(np.array([10, 20, 30], dtype=np.int64), a)

  def testQuantizedTypes(self):
    for dtype in [tf.qint32, tf.quint8, tf.qint8]:
      # Test with array.
      t = tensor_util.make_tensor_proto([(10,), (20,), (30,)], dtype=dtype)
      self.assertEquals(dtype, t.dtype)
      self.assertProtoEquals("dim { size: 3 }", t.tensor_shape)
      self.assertEquals(10, t.int_val[0])
      self.assertEquals(20, t.int_val[1])
      self.assertEquals(30, t.int_val[2])

  def testString(self):
    t = tensor_util.make_tensor_proto("foo")
    self.assertProtoEquals("""
      dtype: DT_STRING
      tensor_shape {}
      string_val: "foo"
      """, t)
    a = tensor_util.MakeNdarray(t)
    self.assertEquals(np.object, a.dtype)
    self.assertEquals([b"foo"], a)

  def testStringWithImplicitRepeat(self):
    t = tensor_util.make_tensor_proto("f", shape=[3, 4])
    a = tensor_util.MakeNdarray(t)
    self.assertAllEqual(np.array([[b"f"] * 4] * 3, dtype=np.object), a)

  def testStringN(self):
    t = tensor_util.make_tensor_proto([b"foo", b"bar", b"baz"], shape=[1, 3])
    self.assertProtoEquals("""
      dtype: DT_STRING
      tensor_shape { dim { size: 1 } dim { size: 3 } }
      string_val: "foo"
      string_val: "bar"
      string_val: "baz"
      """, t)
    a = tensor_util.MakeNdarray(t)
    self.assertEquals(np.object, a.dtype)
    self.assertAllEqual(np.array([[b"foo", b"bar", b"baz"]]), a)

  def testStringNpArray(self):
    t = tensor_util.make_tensor_proto(np.array([[b"a", b"ab"],
                                                [b"abc", b"abcd"]]))
    self.assertProtoEquals("""
      dtype: DT_STRING
      tensor_shape { dim { size: 2 } dim { size: 2 } }
      string_val: "a"
      string_val: "ab"
      string_val: "abc"
      string_val: "abcd"
      """, t)
    a = tensor_util.MakeNdarray(t)
    self.assertEquals(np.object, a.dtype)
    self.assertAllEqual(np.array([[b"a", b"ab"], [b"abc", b"abcd"]]), a)

  def testComplex64(self):
    t = tensor_util.make_tensor_proto((1+2j), dtype=tf.complex64)
    self.assertProtoEquals("""
      dtype: DT_COMPLEX64
      tensor_shape {}
      scomplex_val: 1
      scomplex_val: 2
      """, t)
    a = tensor_util.MakeNdarray(t)
    self.assertEquals(np.complex64, a.dtype)
    self.assertAllEqual(np.array(1 + 2j), a)

  def testComplex128(self):
    t = tensor_util.make_tensor_proto((1+2j), dtype=tf.complex128)
    self.assertProtoEquals("""
      dtype: DT_COMPLEX128
      tensor_shape {}
      dcomplex_val: 1
      dcomplex_val: 2
      """, t)
    a = tensor_util.MakeNdarray(t)
    self.assertEquals(np.complex128, a.dtype)
    self.assertAllEqual(np.array(1 + 2j), a)

  def testComplexWithImplicitRepeat(self):
    for dtype, np_dtype in [(tf.complex64, np.complex64),
                            (tf.complex128, np.complex128)]:
      t = tensor_util.make_tensor_proto((1+1j), shape=[3, 4],
                                        dtype=dtype)
      a = tensor_util.MakeNdarray(t)
      self.assertAllClose(np.array([[(1+1j), (1+1j), (1+1j), (1+1j)],
                                    [(1+1j), (1+1j), (1+1j), (1+1j)],
                                    [(1+1j), (1+1j), (1+1j), (1+1j)]],
                                   dtype=np_dtype), a)

  def testComplex64N(self):
    t = tensor_util.make_tensor_proto([(1+2j), (3+4j), (5+6j)], shape=[1, 3],
                                      dtype=tf.complex64)
    self.assertProtoEquals("""
      dtype: DT_COMPLEX64
      tensor_shape { dim { size: 1 } dim { size: 3 } }
      scomplex_val: 1
      scomplex_val: 2
      scomplex_val: 3
      scomplex_val: 4
      scomplex_val: 5
      scomplex_val: 6
      """, t)
    a = tensor_util.MakeNdarray(t)
    self.assertEquals(np.complex64, a.dtype)
    self.assertAllEqual(np.array([[(1+2j), (3+4j), (5+6j)]]), a)

  def testComplex128N(self):
    t = tensor_util.make_tensor_proto([(1+2j), (3+4j), (5+6j)], shape=[1, 3],
                                      dtype=tf.complex128)
    self.assertProtoEquals("""
      dtype: DT_COMPLEX128
      tensor_shape { dim { size: 1 } dim { size: 3 } }
      dcomplex_val: 1
      dcomplex_val: 2
      dcomplex_val: 3
      dcomplex_val: 4
      dcomplex_val: 5
      dcomplex_val: 6
      """, t)
    a = tensor_util.MakeNdarray(t)
    self.assertEquals(np.complex128, a.dtype)
    self.assertAllEqual(np.array([[(1+2j), (3+4j), (5+6j)]]), a)

  def testComplex64NpArray(self):
    t = tensor_util.make_tensor_proto(
        np.array([[(1+2j), (3+4j)], [(5+6j), (7+8j)]]), dtype=tf.complex64)
    # scomplex_val are real_0, imag_0, real_1, imag_1, ...
    self.assertProtoEquals("""
      dtype: DT_COMPLEX64
      tensor_shape { dim { size: 2 } dim { size: 2 } }
      scomplex_val: 1
      scomplex_val: 2
      scomplex_val: 3
      scomplex_val: 4
      scomplex_val: 5
      scomplex_val: 6
      scomplex_val: 7
      scomplex_val: 8
      """, t)
    a = tensor_util.MakeNdarray(t)
    self.assertEquals(np.complex64, a.dtype)
    self.assertAllEqual(np.array([[(1+2j), (3+4j)], [(5+6j), (7+8j)]]), a)

  def testComplex128NpArray(self):
    t = tensor_util.make_tensor_proto(
        np.array([[(1+2j), (3+4j)], [(5+6j), (7+8j)]]), dtype=tf.complex128)
    # scomplex_val are real_0, imag_0, real_1, imag_1, ...
    self.assertProtoEquals("""
      dtype: DT_COMPLEX128
      tensor_shape { dim { size: 2 } dim { size: 2 } }
      dcomplex_val: 1
      dcomplex_val: 2
      dcomplex_val: 3
      dcomplex_val: 4
      dcomplex_val: 5
      dcomplex_val: 6
      dcomplex_val: 7
      dcomplex_val: 8
      """, t)
    a = tensor_util.MakeNdarray(t)
    self.assertEquals(np.complex128, a.dtype)
    self.assertAllEqual(np.array([[(1+2j), (3+4j)], [(5+6j), (7+8j)]]), a)

  def testUnsupportedDType(self):
    with self.assertRaises(TypeError):
      tensor_util.make_tensor_proto(np.array([1]), 0)

  def testShapeTooLarge(self):
    with self.assertRaises(ValueError):
      tensor_util.make_tensor_proto(np.array([1, 2]), shape=[1])

  def testLowRankSupported(self):
    t = tensor_util.make_tensor_proto(np.array(7))
    self.assertProtoEquals("""
      dtype: DT_INT64
      tensor_shape {}
      int64_val: 7
      """, t)

  def testShapeEquals(self):
    t = tensor_util.make_tensor_proto([10, 20, 30, 40], shape=[2, 2])
    self.assertTrue(tensor_util.ShapeEquals(t, [2, 2]))
    self.assertTrue(tensor_util.ShapeEquals(t, (2, 2)))
    self.assertTrue(tensor_util.ShapeEquals(
        t, tensor_shape.as_shape([2, 2]).as_proto()))
    self.assertFalse(tensor_util.ShapeEquals(t, [5, 3]))
    self.assertFalse(tensor_util.ShapeEquals(t, [1, 4]))
    self.assertFalse(tensor_util.ShapeEquals(t, [4]))


class ConstantValueTest(tf.test.TestCase):

  def testConstant(self):
    np_val = np.random.rand(3, 4, 7).astype(np.float32)
    tf_val = tf.constant(np_val)
    self.assertAllClose(np_val, tf.contrib.util.constant_value(tf_val))

    np_val = np.random.rand(3, 0, 7).astype(np.float32)
    tf_val = tf.constant(np_val)
    self.assertAllClose(np_val, tf.contrib.util.constant_value(tf_val))

  def testUnknown(self):
    tf_val = state_ops.variable_op(shape=[3, 4, 7], dtype=tf.float32)
    self.assertIs(None, tf.contrib.util.constant_value(tf_val))

  def testShape(self):
    np_val = np.array([1, 2, 3], dtype=np.int32)
    tf_val = tf.shape(tf.constant(0.0, shape=[1, 2, 3]))
    c_val = tf.contrib.util.constant_value(tf_val)
    self.assertAllEqual(np_val, c_val)
    self.assertEqual(np.int32, c_val.dtype)

  def testSize(self):
    tf_val = tf.size(tf.constant(0.0, shape=[1, 2, 3]))
    c_val = tf.contrib.util.constant_value(tf_val)
    self.assertEqual(6, c_val)

  def testSizeOfScalar(self):
    tf_val = tf.size(tf.constant(0.0))
    c_val = tf.contrib.util.constant_value(tf_val)
    self.assertEqual(1, c_val)
    self.assertEqual(np.int32, type(c_val))

  def testRank(self):
    tf_val = tf.rank(tf.constant(0.0, shape=[1, 2, 3]))
    c_val = tf.contrib.util.constant_value(tf_val)
    self.assertEqual(3, c_val)

  def testCast(self):
    np_val = np.random.rand(3, 4, 7).astype(np.float32)
    tf_val = tf.cast(tf.constant(np_val), tf.float64)
    c_val = tf.contrib.util.constant_value(tf_val)
    self.assertAllClose(np_val.astype(np.float64), c_val)

    np_val = np.random.rand(3, 0, 7).astype(np.float32)
    tf_val = tf.cast(tf.constant(np_val), tf.float64)
    c_val = tf.contrib.util.constant_value(tf_val)
    self.assertAllClose(np_val.astype(np.float64), c_val)

  def testConcat(self):
    np_val = np.random.rand(3, 4, 7).astype(np.float32)
    tf_val = tf.concat(
        0, [np_val[0:1, :, :], np_val[1:2, :, :], np_val[2:3, :, :]])
    c_val = tf.contrib.util.constant_value(tf_val)
    self.assertAllClose(np_val, c_val)

    tf_val = tf.concat(
        tf.placeholder(tf.int32),
        [np_val[0, :, :], np_val[1, :, :], np_val[2, :, :]])
    c_val = tf.contrib.util.constant_value(tf_val)
    self.assertIs(None, c_val)

    tf_val = tf.concat(
        1,
        [np_val[0, :, :], tf.placeholder(tf.float32),
         np_val[2, :, :]])
    c_val = tf.contrib.util.constant_value(tf_val)
    self.assertIs(None, c_val)


if __name__ == "__main__":
  tf.test.main()
