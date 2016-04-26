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

#ifndef TENSORFLOW_UTIL_TENSOR_FORMAT_H_
#define TENSORFLOW_UTIL_TENSOR_FORMAT_H_

#include <vector>

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

enum TensorFormat {
  FORMAT_NHWC = 0,
  FORMAT_NCHW = 1,
};

// Parse tensor format from the given string.
// Return true if the parsing succeeds, and false if it fails.
bool FormatFromString(const string& format_str, TensorFormat* format);

// Convert a tensor format into string.
string ToString(TensorFormat format);

// Return the position index from a format given a dimension specification with
// a char.
inline int32 GetTensorDimIndex(TensorFormat format, char dimension) {
  if (format == FORMAT_NHWC) {
    switch (dimension) {
      case 'N':
        return 0;
      case 'H':
        return 1;
      case 'W':
        return 2;
      case 'C':
        return 3;
      default:
        LOG(FATAL) << "Invalid dimension: " << dimension;
    }
  } else if (format == FORMAT_NCHW) {
    switch (dimension) {
      case 'N':
        return 0;
      case 'C':
        return 1;
      case 'H':
        return 2;
      case 'W':
        return 3;
      default:
        LOG(FATAL) << "Invalid dimension: " << dimension;
    }
  } else {
    LOG(FATAL) << "Invalid format: " << static_cast<int>(format);
  }
}

// Return the given tensor dimension from a tensor. The tensor is interpretted
// using the specified format, and a dimension specification using a char.
inline int64 GetTensorDim(const Tensor& tensor, TensorFormat format,
                          char dimension) {
  int index = GetTensorDimIndex(format, dimension);
  CHECK(index >= 0 && index < tensor.dims())
      << "Invalid index from the dimension: " << index << ", " << format << ", "
      << dimension;
  return tensor.dim_size(index);
}

// Return the given tensor dimension from a vector that represents the
// dimensions of a tensor.
// The tensor is interpretted using the specified format, and a dimension
// specification using a char.
inline int64 GetTensorDim(const TensorShape& tensor_shape, TensorFormat format,
                          char dimension) {
  int index = GetTensorDimIndex(format, dimension);
  CHECK(index >= 0 && index < tensor_shape.dims())
      << "Invalid index from the dimension: " << index << ", " << format << ", "
      << dimension;
  return tensor_shape.dim_size(index);
}

// Return the given tensor dimension from a tensor shape.
// The tensor is interpretted using the specified format, and a dimension
// specification using a char.
template <typename T>
T GetTensorDim(const std::vector<T>& attributes, TensorFormat format,
               char dimension) {
  int index = GetTensorDimIndex(format, dimension);
  CHECK(index >= 0 && index < attributes.size())
      << "Invalid index from the dimension: " << index << ", " << format << ", "
      << dimension;
  return attributes[index];
}

// Return the string that specifies the data format for convnet operations.
string GetConvnetDataFormatAttrString();

// Return a tensor shape from the given format, and tensor dimensions.
inline TensorShape ShapeFromFormat(TensorFormat format, int64 N, int64 H,
                                   int64 W, int64 C) {
  std::vector<int64> dim_sizes(4);
  dim_sizes[GetTensorDimIndex(format, 'N')] = N;
  dim_sizes[GetTensorDimIndex(format, 'H')] = H;
  dim_sizes[GetTensorDimIndex(format, 'W')] = W;
  dim_sizes[GetTensorDimIndex(format, 'C')] = C;
  return TensorShape(dim_sizes);
}

// Return a tensor shape from the given format, and tensor dimensions.
inline TensorShape ShapeFromFormat(TensorFormat dst_format,
                                   const TensorShape& src_shape,
                                   TensorFormat src_format) {
  if (src_format == dst_format) {
    return src_shape;
  }
  std::vector<int64> dim_sizes(4);
  dim_sizes[GetTensorDimIndex(dst_format, 'N')] =
      GetTensorDim(src_shape, src_format, 'N');
  dim_sizes[GetTensorDimIndex(dst_format, 'H')] =
      GetTensorDim(src_shape, src_format, 'H');
  dim_sizes[GetTensorDimIndex(dst_format, 'W')] =
      GetTensorDim(src_shape, src_format, 'W');
  dim_sizes[GetTensorDimIndex(dst_format, 'C')] =
      GetTensorDim(src_shape, src_format, 'C');
  return TensorShape(dim_sizes);
}

}  // namespace tensorflow

#endif  // TENSORFLOW_UTIL_TENSOR_FORMAT_H_
