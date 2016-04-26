/* Copyright 2015 Google Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TENSORFLOW_KERNELS_LINALG_OPS_COMMON_H_
#define TENSORFLOW_KERNELS_LINALG_OPS_COMMON_H_

// Classes to support linear algebra functionality, similar to the numpy.linalg
// module. Supports batch computation on several matrices at once, sharding the
// computations across different threads if necessary.

#define EIGEN_USE_THREADS

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/kernel_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/work_sharder.h"

namespace tensorflow {

// Base class for unary linear algebra operators.
class UnaryLinearAlgebraOpBase : public OpKernel {
 public:
  explicit UnaryLinearAlgebraOpBase(OpKernelConstruction* context)
      : OpKernel(context) {}
  ~UnaryLinearAlgebraOpBase() override {}

  // Return the output shape of each individual matrix operation. Must be
  // rank 0, 1, or 2.  Scalar outputs are rank 0.
  virtual TensorShape GetOutputMatrixShape(
      const TensorShape& input_matrix_shape) = 0;

  // Return the cost per matrix operation. Cost per unit is assumed to be
  // roughly 1ns, based on comments in core/util/work_sharder.cc.
  virtual int64 GetCostPerUnit(const TensorShape& input_matrix_shape) = 0;

  // If SupportsBatchOperation() returns false, this Op will only accept rank 2
  // (if the supported input type is a matrix). If it returns true, the Op will
  // accept inputs of rank >= 3, and repeatedly execute the operation on all
  // matrices in the innermost two dimensions.
  virtual bool SupportsBatchOperation() = 0;

  // Perform the actual computation on an input matrix, and store the results
  // in the output. This will be called repeatedly for a single call to
  // Compute(), if multiple matrices exist in the input Tensor.
  //
  // This function should only compute the results for a single input matrix.
  // The 'matrix_index' parameter specifies the index of the matrix to be used
  // from the input, and the index of the matrix to be written to in the output.
  // The input matrix is in row major order, and is located at the memory
  // address
  //   in.flat<Scalar>().data() +
  //   matrix_index * input_matrix_shape.num_elements().
  // The output matrix is in row major order, and is located at the memory
  // address
  //   out->flat<Scalar>().data() +
  //   matrix_index * output_matrix_shape.num_elements().
  // The UnaryLinearAlgebraOp<Scalar> class below has functionality which
  // performs
  // this mapping and presents an interface based on the Eigen::MatrixBase API.
  virtual void ComputeMatrix(OpKernelContext* context, int64 matrix_index,
                             const Tensor& in,
                             const TensorShape& input_matrix_shape, Tensor* out,
                             const TensorShape& output_matrix_shape) = 0;

  void Compute(OpKernelContext* context) override;
};

// This base class encapsulates the functionality of mapping the input and
// output tensors using Eigen::Map, so that the Eigen::MatrixBase API may be
// directly used by derived classes.
// SupportsBatchOperationT is a bool template argument which if set to true
// will allow the Op to process batches of matrices (rank >= 3); if set to
// false the Op will only accept rank 2 inputs.
template <typename Scalar, bool SupportsBatchOperationT>
class UnaryLinearAlgebraOp : public UnaryLinearAlgebraOpBase {
 public:
  explicit UnaryLinearAlgebraOp(OpKernelConstruction* context)
      : UnaryLinearAlgebraOpBase(context) {}

  using Matrix =
      Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  using ConstMatrixMap = Eigen::Map<const Matrix>;
  using MatrixMap = Eigen::Map<Matrix>;

  // Perform the actual computation on the input matrix, and store the results
  // in the output. This will be called repeatedly for a single call to
  // Compute(), if multiple matrices exist in the input Tensor.
  virtual void ComputeMatrix(OpKernelContext* context,
                             const ConstMatrixMap& input,
                             MatrixMap* output) = 0;

  bool SupportsBatchOperation() final { return SupportsBatchOperationT; }

  // A concrete implementation of UnaryLinearAlgebraOpBase::ComputeMatrix().
  void ComputeMatrix(OpKernelContext* context, int64 matrix_index,
                     const Tensor& in, const TensorShape& input_matrix_shape,
                     Tensor* out, const TensorShape& output_matrix_shape) final;
};

// Declare that UnaryLinearAlgebraOp is explicitly instantiated in
// linalg_ops_common.cc for float and double.
extern template class UnaryLinearAlgebraOp<float, false>;
extern template class UnaryLinearAlgebraOp<float, true>;
extern template class UnaryLinearAlgebraOp<double, false>;
extern template class UnaryLinearAlgebraOp<double, true>;

}  // namespace tensorflow

#define REGISTER_LINALG_OP(OpName, OpClass, Scalar) \
  REGISTER_KERNEL_BUILDER(                          \
      Name(OpName).Device(DEVICE_CPU).TypeConstraint<Scalar>("T"), OpClass)

#endif  // TENSORFLOW_KERNELS_LINALG_OPS_COMMON_H_
