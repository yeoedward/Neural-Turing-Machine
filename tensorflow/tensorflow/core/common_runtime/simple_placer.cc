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

#include "tensorflow/core/common_runtime/simple_placer.h"

#include <memory>
#include <utility>
#include <vector>

#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/framework/device_attributes.pb.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/stringpiece.h"

namespace tensorflow {

namespace {

// Returns a list of devices sorted by name from 'devices' whose type is in
// 'supported_device_types'.  This function searches in order of the device
// types in 'supported_device_types' and returns the *first* subset of devices
// that match.
//
// For example, if supported_device_types contains {GPU, CPU} and
// 'devices' contains CPU and GPU devices, the returned vector will
// include *only* GPU devices, since that is higher in the priority
// order in 'supported_device_types'.
std::vector<Device*> FilterSupportedDevices(
    const std::vector<Device*>& devices,
    const DeviceTypeVector& supported_device_types) {
  std::vector<Device*> filtered_devices;
  auto device_sort = [](const Device* a, const Device* b) {
    return a->name() < b->name();
  };
  for (DeviceType d : supported_device_types) {
    for (Device* device : devices) {
      if (DeviceType(device->attributes().device_type()) == d) {
        filtered_devices.emplace_back(device);
      }
    }

    // If there are any devices under this device type, return this
    // subset.
    if (!filtered_devices.empty()) {
      std::sort(filtered_devices.begin(), filtered_devices.end(), device_sort);
      return filtered_devices;
    }
  }

  std::sort(filtered_devices.begin(), filtered_devices.end(), device_sort);
  return filtered_devices;
}

// TODO(vrv): Remove "@" syntax capability.
bool HasColocatedNodeName(const Node& node) {
  return StringPiece(node.def().device()).starts_with("@");
}

Status ParseColocatedNodeName(const Node& node,
                              string* out_colocated_node_name) {
  StringPiece device(node.def().device());
  if (!device.Consume("@")) {
    return errors::InvalidArgument("Malformed colocated node name: '", device,
                                   "'");
  }
  // TODO(mrry): Validate that the node name is a valid node name.
  *out_colocated_node_name = device.ToString();
  return Status::OK();
}

// Returns the name of the colocation group of the node by inspecting
// the "_class" attribute of the NodeDef.  Returns "" if it doesn't
// exist.
Status ColocationGroups(const Node& node,
                        std::vector<string>* colocation_groups) {
  std::vector<string> class_specs;
  // TODO(vrv): We should consider adding a GetNodeAttr that returns a
  // StringPiece, to avoid a copy.
  Status s = GetNodeAttr(node.def(), "_class", &class_specs);
  if (!s.ok()) {
    // No "_class" attribute is equivalent to the empty colocation_group.
    *colocation_groups = {strings::StrCat("loc:@", node.name())};
    return Status::OK();
  }

  bool found_spec = false;
  for (const string& class_spec : class_specs) {
    StringPiece spec(class_spec);
    if (spec.Consume("loc:@")) {
      found_spec = true;
      colocation_groups->emplace_back(class_spec);
    }
  }

  if (!found_spec) {
    *colocation_groups = {strings::StrCat("loc:@", node.name())};
  }
  return Status::OK();
}

// This class maintains the connected components of a colocation
// constraint graph, and uses this information to assign a satisfying
// device placement to the nodes of the graph.
//
// The typical usage pattern is:
//
//   Graph graph = ...;
//   DeviceSet device_set = ...;
//   ColocationGraph colocation_graph(graph, device_set);
//
//   // Add all the nodes of graph to colocation_graph.
//   for (Node* node : graph.nodes()) {
//     TF_RETURN_IF_ERROR(colocation_graph.AddNode(*node));
//   }
//
//   // Add one or more colocation constraint.
//   Node node_1 = *graph.FindNodeId(...);
//   Node node_2 = *graph.FindNodeId(...);
//   TF_RETURN_IF_ERROR(colocation_graph.ColocateNodes(node_1, node_2));
//
//   // Assign devices based on the accumulated constraints.
//   for (Node* node : graph.nodes()) {
//     TF_RETURN_IF_ERROR(colocation_graph.AssignDevice(node));
//   }
//
// The implementation uses the union-find algorithm to maintain the
// connected components efficiently and incrementally as edges
// (implied by ColocationGraph::ColocateNodes() invocations) are added.
class ColocationGraph {
 public:
  ColocationGraph(Graph* graph, const DeviceSet* device_set,
                  const SessionOptions* options)
      : device_set_(device_set),
        device_types_(device_set->PrioritizedDeviceTypeList()),
        options_(options) {
    members_.reserve(graph->num_node_ids());
  }

  // Adds the given node to this ColocationGraph as a singleton.
  //
  // NOTE: The implementation assumes that the ids of nodes passed to
  // this method are dense and zero-based; the memory used will be linear in
  // the largest node ID.
  // NOTE: If this method returns an error, *this is left in an undefined
  // state.
  Status AddNode(const Node& node) {
    Member member;
    TF_RETURN_IF_ERROR(InitializeMember(node, &member));
    CHECK_GE(member.parent, 0);
    members_.resize(member.parent + 1);
    members_[member.parent] = std::move(member);

    // When adding the node, identify whether it is part of a
    // colocation group.
    std::vector<string> colocation_groups;
    TF_RETURN_IF_ERROR(ColocationGroups(node, &colocation_groups));
    Status s;
    for (const string& colocation_group : colocation_groups) {
      auto it = colocation_group_root_.find(colocation_group);
      if (it == colocation_group_root_.end()) {
        // This is the first node of the colocation group, so
        // designate this node as the 'root' of that colocation group.
        colocation_group_root_[colocation_group] = &node;
      } else {
        // Try to colocate the node with the root.  If there is an
        // error, return it.
        s = ColocateNodes(node, *(it->second));
        if (!s.ok()) {
          return s;
        }
      }
    }

    return Status::OK();
  }

  // Merge the (possibly disjoint) sets containing nodes "x" and
  // "y". Returns OK if the all nodes in the union of these sets can
  // be placed on the same device type.
  //
  // NOTE: If this method returns an error, *this is left in an undefined
  // state.
  Status ColocateNodes(const Node& x, const Node& y) {
    int x_root = FindRoot(x.id());
    int y_root = FindRoot(y.id());
    Status s;
    if (x_root != y_root) {
      // Merge the sets by swinging the parent pointer of the smaller
      // tree to point to the root of the larger tree. Together with
      // path compression in ColocationGraph::FindRoot, this ensures
      // that we do not experience pathological performance on graphs
      // such as chains.
      int new_root, old_root;
      if (members_[x_root].rank < members_[y_root].rank) {
        // The tree rooted at x_root is shallower, so connect it to
        // y_root. The rank of y_root is unchanged because its new
        // child has strictly less rank.
        members_[x_root].parent = y_root;
        new_root = y_root;
        old_root = x_root;
      } else if (members_[x_root].rank > members_[y_root].rank) {
        // The tree rooted at y_root is shallower, so connect it to
        // x_root. The rank of x_root is unchanged because its new
        // child has strictly less rank.
        members_[y_root].parent = x_root;
        new_root = x_root;
        old_root = y_root;
      } else {
        // Both trees have the same rank, so break the tie by choosing
        // x_root as the new root.
        members_[y_root].parent = x_root;
        // Increment the rank of the tree rooted at x_root, because it
        // is now strictly deeper than before.
        ++members_[x_root].rank;
        new_root = x_root;
        old_root = y_root;
      }

      // Merge the partial device specifications, and ensure that they are
      // compatible. NULL options_ is treated as allowing soft placement.
      // TODO(mrry): Consider enriching the error message by pointing
      // out which nodes have the explicit partial device
      // specifications that caused this conflict.
      s = DeviceNameUtils::MergeDevNames(
          &members_[new_root].device_name, members_[old_root].device_name,
          options_ == nullptr || options_->config.allow_soft_placement());
      if (!s.ok()) {
        return errors::InvalidArgument("Cannot colocate nodes '", x.name(),
                                       "' and '", y.name(), ": ",
                                       s.error_message());
      }

      // Ensure that the common root has at least one supported device
      // type, by computing the intersection of
      // members_[new_root].supported_device_types and
      // members_[old_root].supported_device_types.
      MergeSupportedDevices(&members_[new_root].supported_device_types,
                            members_[old_root].supported_device_types);
      if (members_[x_root].supported_device_types.size() == 0) {
        return errors::InvalidArgument(
            "Cannot colocate nodes '", x.name(), "' and '", y.name(),
            "' because no device type supports both of those nodes and the "
            "other nodes colocated with them");
      }
    }
    return Status::OK();
  }

  // For the given node, subject to the constraints previously given
  // to this ColocationGraph, set its assigned_device_name. Returns OK
  // if a satisfying device can be found, otherwise an error.
  Status AssignDevice(Node* node) {
    int node_root = FindRoot(node->id());
    if (members_[node_root].assigned_device == nullptr) {
      // We have not yet assigned a device for the colocated node set containing
      // n, so we do so now using the constraints on the root node.

      // "devices" will contain the set of feasible placements for the
      // colocated node set containing n.
      std::vector<Device*> devices;
      if (DeviceNameUtils::HasSomeDetails(members_[node_root].device_name)) {
        // The root node has a (possibly partial) device
        // specification, so enumerate the physical devices that
        // conform to it.
        device_set_->FindMatchingDevices(members_[node_root].device_name,
                                         &devices);

        if (!devices.empty()) {
          // Filter devices into those that are compatible with the root
          // node (and its children).
          devices = FilterSupportedDevices(
              devices, members_[node_root].supported_device_types);
        }

        // Perform soft placement if allow_soft_placement is set.  options_
        // being NULL is treated as allowing soft placement.
        if (devices.empty() &&
            (options_ == nullptr || options_->config.allow_soft_placement())) {
          // The soft_device_name is the same as the node's device name
          // without specifying the device type or ID.
          DeviceNameUtils::ParsedName soft_device_name =
              members_[node_root].device_name;
          soft_device_name.type.clear();
          soft_device_name.has_type = false;
          soft_device_name.has_id = false;
          device_set_->FindMatchingDevices(soft_device_name, &devices);
          if (!devices.empty()) {
            devices = FilterSupportedDevices(
                devices, members_[node_root].supported_device_types);
          }
        }

        if (devices.empty()) {
          // Return an error when a physical device that matches an explicit
          // device specification is not found. This ensures that we don't
          // assign a node to GPU when the user wanted to force it on CPU.
          DeviceNameUtils::ParsedName specified_device_name;
          if (DeviceNameUtils::ParseFullName(node->def().device(),
                                             &specified_device_name) &&
              specified_device_name == members_[node_root].device_name) {
            // The specified device and merged set device match, and
            // will appear in the GraphDef (for debugging), so just
            // print the specified device.
            std::vector<Device*> devices_matching_nodedef;
            device_set_->FindMatchingDevices(specified_device_name,
                                             &devices_matching_nodedef);
            if (devices_matching_nodedef.empty()) {
              // Sometimes it is almost impossible to understand the problem
              // without a list of available devices.
              std::vector<string> device_names;
              for (const Device* device : device_set_->devices()) {
                device_names.push_back(device->name());
              }
              std::sort(device_names.begin(), device_names.end());

              return errors::InvalidArgument(
                  "Could not satisfy explicit device specification '",
                  node->def().device(),
                  "' because no devices matching that specification "
                  "are registered in this process; available devices: ",
                  str_util::Join(device_names, ", "));
            } else if (specified_device_name.has_type) {
              return errors::InvalidArgument(
                  "Could not satisfy explicit device specification '",
                  node->def().device(), "' because no supported kernel for ",
                  specified_device_name.type, " devices is available");
            } else {
              return errors::InvalidArgument(
                  "Could not satisfy explicit device specification '",
                  node->def().device());
            }
          } else {
            // The specified device may be a valid device but the
            // merged set device is different, so print both.
            return errors::InvalidArgument(
                "Could not satisfy explicit device specification '",
                node->def().device(),
                "' because the node was colocated with a group of nodes that "
                "required incompatible device '",
                DeviceNameUtils::ParsedNameToString(
                    members_[node_root].device_name),
                "'");
          }
        }
      } else {
        // The device is completely unspecified, so enumerate the devices that
        // support all of the nodes in the set.
        if (device_set_->devices().empty()) {
          return errors::Internal("No devices are registered");
        }
        devices = FilterSupportedDevices(
            device_set_->devices(), members_[node_root].supported_device_types);

        if (devices.empty()) {
          return errors::InvalidArgument(
              "Node had no OpKernel registered to support this operation: ",
              "Operation was ", node->type_string(), " and inputs were ",
              DataTypeVectorString(node->input_types()));
        }
      }

      // Returns the first device in sorted devices list so we will always
      // choose the same device.
      members_[node_root].assigned_device = devices[0];
    }
    node->set_assigned_device_name(members_[node_root].assigned_device->name());

    // Log placement if log_device_placement is set.
    if (options_ && options_->config.log_device_placement()) {
      printf("%s: %s\n", node->name().c_str(),
             node->assigned_device_name().c_str());
      LOG(INFO) << node->name() << ": " << node->assigned_device_name();
    }

    return Status::OK();
  }

 private:
  // Represents a node in the disjoint node set forest, and the
  // accumulated constraints on the device used by that node.
  struct Member {
    Member() = default;
    // The id of the node that is the parent of this one, or its own
    // id if it is a root. parent <= 0 indicates that this member is invalid.
    int parent = -1;
    // A proxy for the depth of the tree that is used to prefer
    // connecting smaller trees to larger trees when merging disjoint
    // sets.
    int rank = 0;
    // The intersection of all device types supported by this node,
    // and those of all of its children, in priority order
    // of the preferred device.
    DeviceTypeVector supported_device_types;
    // The merged form of the device requested for this node, with
    // those of all of its children.
    DeviceNameUtils::ParsedName device_name;
    // If this node is a root, stores the Device to which this node
    // and all of its children have been assigned, or nullptr if this
    // has not yet been computed by GetAssignedDevice().
    Device* assigned_device = nullptr;
  };

  Status InitializeMember(const Node& node, Member* member) {
    const int id = node.id();
    if (id < 0) {
      return errors::InvalidArgument("Node id was not positive: ", id);
    }
    member->parent = id;
    TF_RETURN_IF_ERROR(SupportedDeviceTypesForNode(
        device_types_, node.def(), &member->supported_device_types));

    if (!node.assigned_device_name().empty()) {
      // This node has already been assigned to a device, so we
      // respect this placement, after sanity-checking it.  The
      // device_name and supported_device_types for this node reflect
      // the assigned device, so any nodes colocated with this node
      // will be assigned to the same device (assuming this is
      // possible).
      // NOTE: Since any assignment must have been performed by
      // the TensorFlow runtime, we consider errors in this branch to
      // be INTERNAL.
      if (!DeviceNameUtils::ParseFullName(node.assigned_device_name(),
                                          &member->device_name)) {
        return errors::Internal("Malformed assigned device '",
                                node.assigned_device_name(), "'");
      }
      std::vector<Device*> devices;
      const Device* assigned_device =
          device_set_->FindDeviceByName(node.assigned_device_name());
      if (assigned_device == nullptr) {
        return errors::Internal("Assigned device '",
                                node.assigned_device_name(),
                                "' does not match any device");
      }

      for (DeviceType d : member->supported_device_types) {
        if (DeviceType(assigned_device->attributes().device_type()) == d) {
          return Status::OK();
        }
      }

      return errors::Internal("Assigned device '", node.assigned_device_name(),
                              "' does not have registered OpKernel support "
                              "for ",
                              node.def().op());
    } else {
      // This node has not yet been assigned to a device, so we
      // calculate any constraints due to the set of registered
      // kernels and any (partial) user-provided device specification
      // in the NodeDef.

      // If no kernels are registered for this op type, fail with an error.
      if (member->supported_device_types.empty()) {
        return errors::InvalidArgument(
            "No OpKernel was registered to support "
            "Op '",
            node.def().op(), "' with these attrs");
      }

      // If the NodeDef contains a device that is *not* a colocated node name
      // (i.e. it does not begin with '@') then we interpret it as a (partial)
      // device specification.
      string colocated_node_name;
      if (!node.def().device().empty() && !HasColocatedNodeName(node)) {
        // The user has specified a device in the NodeDef, try to find a
        // valid device matching their specification in the set of
        // devices.
        // NOTE: The full name may specify a device that is not in
        // n.supported_device_types(), but we check that in AssignDevice().
        if (!DeviceNameUtils::ParseFullName(node.def().device(),
                                            &member->device_name)) {
          return errors::InvalidArgument("Malformed device specification '",
                                         node.def().device(), "'");
        }
      }
    }
    return Status::OK();
  }

  // Updates target to contain the intersection of the device types in
  // "target" and "other".
  static void MergeSupportedDevices(DeviceTypeVector* target,
                                    const DeviceTypeVector& other) {
    DeviceTypeVector temp = *target;
    target->clear();

    // Iterate in priority order.
    for (DeviceType device_type : temp) {
      bool found = false;
      for (DeviceType other_device_type : other) {
        if (device_type == other_device_type) {
          found = true;
          break;
        }
      }
      if (found) {
        target->push_back(device_type);
      }
    }
  }

  // Returns the root node of the disjoint tree to which the node with the
  // given id is connected.
  int FindRoot(int node_id) {
    DCHECK_GE(members_[node_id].parent, 0);
    if (members_[node_id].parent != node_id) {
      // NOTE: Compress paths from node_id to its root, so that future
      // calls to FindRoot and ColocateNodes are more efficient.
      members_[node_id].parent = FindRoot(members_[node_id].parent);
    }
    return members_[node_id].parent;
  }

  std::vector<Member> members_;
  const DeviceSet* device_set_;  // Not owned.
  const std::vector<DeviceType> device_types_;
  const SessionOptions* options_;  // Not owned;

  // Maps from a colocation group identifier to the 'root' of that
  // colocation group.
  std::unordered_map<string, const Node*> colocation_group_root_;
};

}  // namespace

SimplePlacer::SimplePlacer(Graph* graph, const DeviceSet* devices,
                           const NodeNameToIdMap* name_to_id_map,
                           const SessionOptions* options)
    : graph_(graph),
      devices_(devices),
      name_to_id_map_(name_to_id_map),
      options_(options) {}

SimplePlacer::SimplePlacer(Graph* graph, const DeviceSet* devices,
                           const NodeNameToIdMap* name_to_id_map)
    : graph_(graph), devices_(devices), name_to_id_map_(name_to_id_map) {
  options_ = nullptr;
}

SimplePlacer::~SimplePlacer() {}

Status SimplePlacer::Run() {
  if (devices_->devices().empty()) {
    return errors::FailedPrecondition("No devices are registered");
  }

  ColocationGraph colocation_graph(graph_, devices_, options_);
  Status status;

  // 1. First add all of the nodes. Note that steps (1) and (2)
  // requires two passes over the nodes because the graph (and hence
  // the constraints) may not be acyclic.
  for (Node* node : graph_->nodes()) {
    // Skip the source and sink nodes.
    if (!node->IsOp()) {
      continue;
    }
    status = colocation_graph.AddNode(*node);
    if (!status.ok()) return AttachDef(status, node->def());
  }

  // 2. Enumerate the constraint edges, and use them to update the disjoint
  // node set.
  for (Node* node : graph_->nodes()) {
    if (!node->IsOp()) {
      continue;
    }

    // 2(a). If node n specifies a colocation constraint as its device name,
    // add an edge from the colocated node to n.
    if (HasColocatedNodeName(*node)) {
      string colocated_node_name;
      status = ParseColocatedNodeName(*node, &colocated_node_name);
      if (!status.ok()) {
        return AttachDef(status, node->def());
      }
      Node* colocated_node;
      status = GetNodeByName(colocated_node_name, &colocated_node);
      if (!status.ok()) {
        return AttachDef(
            errors::InvalidArgument("Colocated node named in device '",
                                    colocated_node_name, "' does not exist"),
            node->def());
      }
      status = colocation_graph.ColocateNodes(*colocated_node, *node);
      if (!status.ok()) {
        return AttachDef(
            errors::InvalidArgument(
                "Cannot satisfy colocation constraint named in device '",
                colocated_node_name, "': ", status.error_message()),
            node->def());
      }
    }

    // 2(b). If `node` has an input edge with reference type, add an
    // edge from the source of that edge to `node`.
    for (const auto& edge : node->in_edges()) {
      if (!edge->IsControlEdge() &&
          IsRefType(node->input_type(edge->dst_input()))) {
        status = colocation_graph.ColocateNodes(*edge->src(), *node);
        if (!status.ok()) {
          return AttachDef(errors::InvalidArgument(
                               "Nodes were connected by a "
                               "reference connection (requiring them to "
                               "be on the same device), but the two nodes "
                               "were assigned two different devices: ",
                               status.error_message()),
                           node->def());
        }
      }
    }
  }

  // 3. For each node, assign a device based on the constraints in the
  // disjoint node set.
  for (Node* node : graph_->nodes()) {
    // Skip the source and sink nodes.
    if (!node->IsOp()) {
      continue;
    }
    // Skip nodes that already have an assigned name.
    if (!node->assigned_device_name().empty()) {
      continue;
    }

    status = colocation_graph.AssignDevice(node);
    if (!status.ok()) {
      return AttachDef(
          errors::InvalidArgument("Cannot assign a device to node '",
                                  node->name(), "': ", status.error_message()),
          node->def());
    }
  }
  return Status::OK();
}

Status SimplePlacer::GetNodeByName(const string& name, Node** out_node) const {
  NodeNameToIdMap::const_iterator iter = name_to_id_map_->find(name);
  if (iter != name_to_id_map_->end()) {
    *out_node = graph_->FindNodeId(iter->second);
    if (*out_node) {
      return Status::OK();
    }
  }
  return errors::NotFound(name);
}

}  // namespace tensorflow
