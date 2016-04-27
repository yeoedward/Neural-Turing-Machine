#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

REGISTER_OP("NTMRotate")
  .Input("weights: float")
  .Input("shifts: float")
  .Output("shifted_weights: float");

using namespace tensorflow;

class NTMRotateOp : public OpKernel {
 public:
  explicit NTMRotateOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the input tensors
    const Tensor& weights_tensor = context->input(0);
    auto weights = weights_tensor.flat<float>();
    const Tensor& shifts_tensor = context->input(1);
    auto shifts = shifts_tensor.flat<float>();

    const TensorShape& weights_shape = weights_tensor.shape();
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

    // Create an output tensor
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, weights_tensor.shape(),
                                                     &output_tensor));
    auto output = output_tensor->template flat<float>();

    for (int i = 0; i < nbatches; i++) {
      for (int j = 0; j < nrows; j++) {
        float total = 0;
        for (int k = 0; k < nrows; k++) {
          int shift_idx = (((j - k) % nshift) + nshift) % nshift;
          total += weights(i*nrows + k) * shifts(i*nshift + shift_idx);
        }
        output(i*nrows + j) = total;
      }
    }
  }
};

REGISTER_KERNEL_BUILDER(Name("NTMRotate").Device(DEVICE_CPU), NTMRotateOp);
