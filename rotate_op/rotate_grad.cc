#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

REGISTER_OP("NTMRotateGrad")
  .Input("weights: float")
  .Input("shifts: float")
  .Input("grad: float")
  .Output("weights_grad: float")
  .Output("shifts_grad: float");

using namespace tensorflow;

class NTMRotateGradOp : public OpKernel {
 public:
  explicit NTMRotateGradOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the input tensors
    const Tensor& grad_tensor = context->input(0);
    auto grad = grad_tensor.flat<float>();
    const Tensor& weights_tensor = context->input(1);
    auto weights = weights_tensor.flat<float>();
    const Tensor& shifts_tensor = context->input(2);
    auto shifts = shifts_tensor.flat<float>();

    const TensorShape& weights_shape = weights_tensor.shape();
    OP_REQUIRES(context, grad_tensor.shape().IsSameSize(weights_shape),
        errors::InvalidArgument("Gradient tensor should have the same shape"
                                " as weight tensor"));
    OP_REQUIRES(context, weights_shape.dims() == 2,
                errors::InvalidArgument("Weights tensor rank: 2 but got: ",
                                       weights_shape.dims()));
    const int nbatches = weights_shape.dim_size(0);
    const int nrows = weights_shape.dim_size(1);
    const TensorShape& shifts_shape = shifts_tensor.shape();
    OP_REQUIRES(context, shifts_shape.dims() == 2,
                errors::InvalidArgument("Shifts tensor rank: 2 but got: ",
                                       shifts_shape.dims()));
    const int nshift = shifts_tensor.shape().dim_size(1);

    // Create output gradient tensors.
    Tensor* weights_grad_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, weights_tensor.shape(),
                                                     &weights_grad_tensor));
    Tensor* shifts_grad_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, shifts_tensor.shape(),
                                                     &shifts_grad_tensor));
    auto weights_grad = weights_grad_tensor->template flat<float>();
    auto shifts_grad = shifts_grad_tensor->template flat<float>();

    for (int b = 0; b < nbatches; b++) {
      for (int i = 0; i < nrows; i++) {
        float total = 0;
        for (int j = 0; j < nrows; j++) {
          int shift_idx = (((j - i + 1) % nrows) + nrows) % nrows;
          if (shift_idx < nshift) {
            total += grad(b*nrows + j) * shifts(b*nshift + shift_idx);
          }
        }
        weights_grad(b*nrows + i) = total;
      }
    }

    for (int b = 0; b < nbatches; b++) {
      for (int i = 0; i < nshift; i++) {
        float total = 0;
        for (int j = 0; j < nrows; j++) {
          float subtotal = 0;
          for (int k = 0; k < nrows; k++) {
            int shift_idx = (((j - k + 1) % nrows) + nrows) % nrows;
            if (shift_idx == i) {
              subtotal += weights(b*nrows + k);
            }
          }
          total += grad(b*nrows + j) * subtotal;
        }
        shifts_grad(b*nshift + i) = total;
      }
    }
  }
};

REGISTER_KERNEL_BUILDER(
  Name("NTMRotateGrad").Device(DEVICE_CPU),
  NTMRotateGradOp);
