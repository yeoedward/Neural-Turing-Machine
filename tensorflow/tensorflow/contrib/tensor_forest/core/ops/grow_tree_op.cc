// Copyright 2016 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================
// GrowTree adds children to the tree for finished nodes by using the
// end_of_tree tensor as an indicator for where free nodes are in the
// pre-allocated tree tensor.
// For example if the tree is:
//    1, -1, -1, -2, -2, -2, ...
// Then end_of_tree should be 3 (the first -2, or "free" slot in the tensor).
// If node 1 is now finished, the tree tensor after this op would be:
//    1, 3, -1, -1, -1, -2, ...
// and end_of_tree would be 5.

#include "tensorflow/contrib/tensor_forest/core/ops/tree_utils.h"

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/platform/logging.h"


namespace tensorflow {

using tensorforest::CHILDREN_INDEX;
using tensorforest::FEATURE_INDEX;

using tensorforest::LEAF_NODE;


REGISTER_OP("GrowTree")
  .Input("end_of_tree: int32")
  .Input("tree_depths: int32")
  .Input("node_to_accumulator: int32")
  .Input("finished_nodes: int32")
  .Input("best_splits: int32")
  .Input("candidate_split_features: int32")
  .Input("candidate_split_thresholds: float")
  .Output("nodes_to_update: int32")
  .Output("tree_updates: int32")
  .Output("threshold_updates: float")
  .Output("depth_updates: int32")
  .Output("new_end_of_tree: int32")
  .Doc(R"doc(
  Output the tree changes needed to resolve fertile nodes.

  Previous Ops have already decided which fertile nodes want to stop being
  fertile and what their best candidate split should be and have passed that
  information to this Op in `finished_nodes` and `best_splits`.  This Op
  merely checks that there is still space in tree to add new nodes, and if
  so, writes out the sparse updates needed for the fertile nodes to be
  resolved to the tree, threshold and depth tensors.

  end_of_tree: `end_of_tree[0]` is the number of allocated nodes, or
    equivalently the index of the first free node in the tree tensor.
  tree_depths: `tree_depths[i]` is the depth in the tree of node i.
  node_to_accumulator: `node_to_accumulator[i]` is the accumulator slot used by
    fertile node i, or -1 if node i isn't fertile.
  finished_nodes:= A 1-d int32 tensor containing the indices of finished nodes.
  best_splits: `best_splits[i]` is the index of the best split for
    `finished_nodes[i]`.
  candidate_split_features: `candidate_split_features[a][s]` is the feature
    being considered for split s of the fertile node associated with
    accumulator slot a.
  candidate_split_thresholds: `candidate_split_thresholds[a][s]` is the
    threshold value being considered for split s of the fertile node associated
    with accumulator slot a.
  nodes_to_update:= A 1-d int32 tensor containing the node indices that need
    updating.
  tree_updates: The updates to apply to the 2-d tree tensor.  Intended to be
    used with `tf.scatter_update(tree, nodes_to_update, tree_updates)`.
  threshold_updates: The updates to apply to the 1-d thresholds tensor.
    Intended to be used with
    `tf.scatter_update(thresholds, nodes_to_update, threshold_updates)`.
  depth_updates: The updates to apply to the 1-d depths tensor.  Intended to
    be used with `tf.scatter_update(depths, nodes_to_update, depth_updates)`.
  new_end_of_tree: `new_end_of_tree[0]` is the new size of the tree.
)doc");

class GrowTree : public OpKernel {
 public:
  explicit GrowTree(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& end_of_tree = context->input(0);
    const Tensor& tree_depths = context->input(1);
    const Tensor& node_to_accumulator = context->input(2);
    const Tensor& finished = context->input(3);
    const Tensor& best_splits = context->input(4);
    const Tensor& candidate_split_features = context->input(5);
    const Tensor& candidate_split_thresholds = context->input(6);

    OP_REQUIRES(context, end_of_tree.shape().dims() == 1,
                errors::InvalidArgument(
                    "end_of_tree should be one-dimensional"));
    OP_REQUIRES(context, tree_depths.shape().dims() == 1,
                errors::InvalidArgument(
                    "tree_depths should be one-dimensional"));
    OP_REQUIRES(context, node_to_accumulator.shape().dims() == 1,
                errors::InvalidArgument(
                    "node_to_accumulator should be one-dimensional"));
    OP_REQUIRES(context, finished.shape().dims() == 1,
                errors::InvalidArgument(
                    "finished should be one-dimensional"));
    OP_REQUIRES(context, best_splits.shape().dims() == 1,
                errors::InvalidArgument(
                    "best_splits should be one-dimensional"));
    OP_REQUIRES(context, candidate_split_features.shape().dims() == 2,
                errors::InvalidArgument(
                    "candidate_split_features should be two-dimensional"));
    OP_REQUIRES(context, candidate_split_thresholds.shape().dims() == 2,
                errors::InvalidArgument(
                    "candidate_split_thresholds should be two-dimensional"));

    OP_REQUIRES(
        context,
        finished.shape().dim_size(0) ==
        best_splits.shape().dim_size(0),
        errors::InvalidArgument(
            "Number of finished nodes should be the same in finished and "
            "best_splits."));
    OP_REQUIRES(
        context,
        tree_depths.shape().dim_size(0) ==
        node_to_accumulator.shape().dim_size(0),
        errors::InvalidArgument(
            "Number of nodes should be the same in tree_depths and "
            "node_to_accumulator."));
    OP_REQUIRES(
        context,
        candidate_split_features.shape().dim_size(0) ==
        candidate_split_thresholds.shape().dim_size(0),
        errors::InvalidArgument(
            "Number of accumulators should be the same in "
            "candidate_split_features and candidate_split_thresholds."));
    OP_REQUIRES(
        context,
        candidate_split_features.shape().dim_size(1) ==
        candidate_split_thresholds.shape().dim_size(1),
        errors::InvalidArgument(
            "Number of splits should be the same in "
            "candidate_split_features and candidate_split_thresholds."));

    int32 current_end_of_tree = end_of_tree.unaligned_flat<int32>()(0);
    const auto depths = tree_depths.unaligned_flat<int32>();
    const auto node_map = node_to_accumulator.unaligned_flat<int32>();
    const auto finished_vec = finished.unaligned_flat<int32>();
    const auto best_vec = best_splits.unaligned_flat<int32>();
    const auto split_features = candidate_split_features.tensor<int32, 2>();
    const auto split_thresholds = candidate_split_thresholds.tensor<float, 2>();

    const int32 num_finished = finished.shape().dim_size(0);
    const int32 num_nodes = node_to_accumulator.shape().dim_size(0);

    // Converting a leaf node into an internal node requires space for its
    // two children.
    int32 remaining_node_space = (num_nodes - current_end_of_tree) / 2;
    int32 nodes_we_can_allocate = std::min(num_finished, remaining_node_space);
    // Each conversion touches three nodes: the transitioning node and its
    // two new children.
    int32 num_updates = 3 * nodes_we_can_allocate;

    Tensor* nodes_to_update_tensor = nullptr;
    TensorShape nodes_to_update_shape;
    nodes_to_update_shape.AddDim(num_updates);
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, nodes_to_update_shape,
                                            &nodes_to_update_tensor));
    auto nodes_to_update_flat = nodes_to_update_tensor->tensor<int32, 1>();

    Tensor* tree_updates_tensor = nullptr;
    TensorShape tree_updates_shape;
    tree_updates_shape.AddDim(num_updates);
    tree_updates_shape.AddDim(2);
    OP_REQUIRES_OK(context,
                   context->allocate_output(1, tree_updates_shape,
                                            &tree_updates_tensor));
    auto tree_updates_flat = tree_updates_tensor->tensor<int32, 2>();

    Tensor* threshold_updates_tensor = nullptr;
    TensorShape threshold_updates_shape;
    threshold_updates_shape.AddDim(num_updates);
    OP_REQUIRES_OK(context,
                   context->allocate_output(2, threshold_updates_shape,
                                            &threshold_updates_tensor));
    auto threshold_updates_flat = threshold_updates_tensor->tensor<float, 1>();

    Tensor* depth_updates_tensor = nullptr;
    TensorShape depth_updates_shape;
    depth_updates_shape.AddDim(num_updates);
    OP_REQUIRES_OK(context,
                   context->allocate_output(3, depth_updates_shape,
                                            &depth_updates_tensor));
    auto depth_updates_flat = depth_updates_tensor->tensor<int32, 1>();

    int output_slot = 0;
    for (int i = 0; i < nodes_we_can_allocate; i++) {
      const int32 node = finished_vec(i);
      const int32 best = best_vec(i);
      const int32 accumulator = node_map(node);
      if (accumulator < 0) {
        LOG(ERROR) << "Finished node doesn't have an accumulator.";
        continue;
      }

      if (current_end_of_tree >= num_nodes - 1) {
        LOG(ERROR) << "Could not grow tree any further.";
        return;
      }
      const int32 left = current_end_of_tree;
      nodes_to_update_flat(output_slot) = node;

      tree_updates_flat(output_slot, CHILDREN_INDEX) = left;
      tree_updates_flat(output_slot, FEATURE_INDEX) =
          split_features(accumulator, best);
      threshold_updates_flat(output_slot) = split_thresholds(accumulator, best);
      depth_updates_flat(output_slot) = depths(node);
      output_slot++;

      nodes_to_update_flat(output_slot) = left;
      tree_updates_flat(output_slot, CHILDREN_INDEX) = LEAF_NODE;
      tree_updates_flat(output_slot, FEATURE_INDEX) = -1;
      threshold_updates_flat(output_slot) = 0.0;
      depth_updates_flat(output_slot) = depths(node) + 1;
      output_slot++;

      nodes_to_update_flat(output_slot) = left + 1;
      tree_updates_flat(output_slot, CHILDREN_INDEX) = LEAF_NODE;
      tree_updates_flat(output_slot, FEATURE_INDEX) = -1;
      threshold_updates_flat(output_slot) = 0.0;
      depth_updates_flat(output_slot) = depths(node) + 1;
      output_slot++;

      current_end_of_tree += 2;
    }

    Tensor* new_end_of_tree_tensor = nullptr;
    TensorShape new_end_of_tree_shape;
    new_end_of_tree_shape.AddDim(1);
    OP_REQUIRES_OK(context,
                   context->allocate_output(4, new_end_of_tree_shape,
                                            &new_end_of_tree_tensor));
    auto new_end_of_tree_flat = new_end_of_tree_tensor->tensor<int32, 1>();
    new_end_of_tree_flat(0) = current_end_of_tree;
  }
};

REGISTER_KERNEL_BUILDER(Name("GrowTree").Device(DEVICE_CPU), GrowTree);

}  // namespace tensorflow
